import glob
import random
import os
import sys
import numpy as np
import skimage
from PIL import Image
import torch
import torch.nn.functional as F
try:
    from yolov3.utils.augmentations import horisontal_flip
except:
    from utils.augmentations import horisontal_flip
try:
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
except:
    pass
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    import flowiz
except ModuleNotFoundError:
    import traceback
    traceback.print_exc()


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                 more_augment=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.more_augment = more_augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            labels = boxes[:, 0]
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            if self.augment and self.more_augment:
                bbs = ia.BoundingBoxesOnImage(
                    [BoundingBox(x1i, y1i, x2i, y2i, label=l) for x1i, y1i, x2i, y2i, l in zip(x1, y1, x2, y2, labels)],
                    shape=(padded_h, padded_w)
                )
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.9, 1.1), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-10, 10),
                    )
                ], random_order=True)
                orig_bbs = bbs
                img_aug, bbs = seq(image=img.numpy().transpose(1, 2, 0) * 255., bounding_boxes=orig_bbs)
                bbs = bbs.clip_out_of_image()
                while not len(bbs.bounding_boxes) and len(orig_bbs.bounding_boxes):
                    img_aug, bbs = seq(image=img.numpy().transpose(1, 2, 0) * 255., bounding_boxes=orig_bbs)
                    bbs = bbs.remove_out_of_image().clip_out_of_image()
                img = torch.from_numpy(img_aug.transpose(2, 0, 1).copy() / 255.)
                arr = bbs.to_xyxy_array()
                if len(bbs.bounding_boxes):
                    labels = torch.stack(
                    [bbs.bounding_boxes[i].label for i in range(len(bbs.bounding_boxes))])
                else:
                    labels = boxes[:, 0]
                x1 = torch.from_numpy(arr[:, 0])
                x2 = torch.from_numpy(arr[:, 2])
                y1 = torch.from_numpy(arr[:, 1])
                y2 = torch.from_numpy(arr[:, 3])

            # Returns (x, y, w, h)
            boxes = boxes[:len(x1), :]
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            boxes[:, 0] = labels

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ListFlowDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                 more_augment=False, return_flow=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.flow_files = [
            path.replace("images", "optical_flow").replace(".png", "_feature-flow.flo").replace(".jpg", "_feature-flow.flo")
            for path in self.img_files
        ]
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_files = [
            self.img_files[i] for i in range(len(self.flow_files)) if
            os.path.exists(self.flow_files[i].rstrip())
        ]
        self.label_files = [
            self.label_files[i] for i in range(len(self.flow_files)) if
            os.path.exists(self.flow_files[i].rstrip())
        ]
        self.flow_files = [
            self.flow_files[i] for i in range(len(self.flow_files)) if
            os.path.exists(self.flow_files[i].rstrip())
        ]


        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.more_augment = more_augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.return_flow = return_flow

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        flow_path = self.flow_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor as grayscale
        img = Image.open(img_path)

        # extract flow images
        flo = flowiz.read_flow(flow_path)
        flo = skimage.transform.resize(flo, img.size[::-1])
        flo = np.linalg.norm(flo, axis=2)
        flo = np.uint8( (flo / np.max(flo)) ** -1 * 255)
        if not self.return_flow:
            img.putalpha(Image.fromarray(flo, 'L'))
        else:
            flo = transforms.ToTensor()(Image.fromarray(flo, 'L'))
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            labels = boxes[:, 0]
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            if self.augment and self.more_augment:
                bbs = ia.BoundingBoxesOnImage(
                    [BoundingBox(x1i, y1i, x2i, y2i, label=l) for x1i, y1i, x2i, y2i, l in zip(x1, y1, x2, y2, labels)],
                    shape=(padded_h, padded_w)
                )
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.9, 1.1), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-10, 10),
                    )
                ], random_order=True)
                orig_bbs = bbs
                img_aug, bbs = seq(image=img.numpy().transpose(1, 2, 0) * 255., bounding_boxes=orig_bbs)
                bbs = bbs.clip_out_of_image()
                while not len(bbs.bounding_boxes) and len(orig_bbs.bounding_boxes):
                    img_aug, bbs = seq(image=img.numpy().transpose(1, 2, 0) * 255., bounding_boxes=orig_bbs)
                    bbs = bbs.remove_out_of_image().clip_out_of_image()
                img = torch.from_numpy(img_aug.transpose(2, 0, 1).copy() / 255.)
                arr = bbs.to_xyxy_array()
                if len(bbs.bounding_boxes):
                    labels = torch.stack(
                    [bbs.bounding_boxes[i].label for i in range(len(bbs.bounding_boxes))])
                else:
                    labels = boxes[:, 0]
                x1 = torch.from_numpy(arr[:, 0])
                x2 = torch.from_numpy(arr[:, 2])
                y1 = torch.from_numpy(arr[:, 1])
                y2 = torch.from_numpy(arr[:, 3])

            # Returns (x, y, w, h)
            boxes = boxes[:len(x1), :]
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            boxes[:, 0] = labels

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        if self.return_flow:
            return img_path, img, flo, targets
        else:
            return img_path, img, targets

    def collate_fn(self, batch):
        if self.return_flow:
            paths, imgs, flow, targets = list(zip(*batch))
        else:
            paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        if not self.return_flow:
            imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        else:
            imgs = (
                torch.stack([resize(img, self.img_size) for img in imgs]),
                torch.stack([resize(img, self.img_size) for img in flow])
            )
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

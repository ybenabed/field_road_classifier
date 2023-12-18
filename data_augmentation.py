import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.transforms import (
    RandomApply,
    RandomChoice,
    RandomRotation,
    GaussianBlur,
    RandomAutocontrast,
)
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import copy


def addnoise(input_image, noise_factor=0.3):
    inputs = transforms.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0, 1.0)
    output_image = transforms.ToPILImage()
    image = output_image(noisy)
    return image


def augment_image(img_path):
    # orig_image
    orig_img = Image.open(Path(img_path))

    # grayscale
    grayscale_transform = transforms.Grayscale(3)
    grayscaled_image = grayscale_transform(orig_img)

    # horizontal rotation
    horizontal_rotation_transformation = transforms.RandomRotation(180)
    horizontal_rotation_image = horizontal_rotation_transformation(orig_img)

    # Gausian Noise
    gausian_image_3 = addnoise(orig_img)
    gausian_image_6 = addnoise(orig_img, 0.6)

    # Color Jitter
    colour_jitter_transformation = transforms.ColorJitter(
        brightness=(0.5, 1.5), contrast=(2), saturation=(1.4), hue=(-0.1, 0.5)
    )
    colour_jitter_image = colour_jitter_transformation(orig_img)

    return [
        orig_img,
        grayscaled_image,
        horizontal_rotation_image,
        gausian_image_3,
        gausian_image_6,
        colour_jitter_image,
    ]


def creating_file_with_augmented_images(master_dataset_folder, augmented_images_folder):
    files_in_master_dataset = os.listdir(master_dataset_folder)
    try:
        os.makedirs(f"{augmented_images_folder}")
    except FileExistsError:
        pass

    for counter, element in enumerate(files_in_master_dataset):
        counter = counter + 1
        augmented_images = augment_image(os.path.join(master_dataset_folder, element))
        for counter_2, augmented_image in enumerate(augmented_images):
            augmented_image = augmented_image.save(
                f"{augmented_images_folder}/{counter_2}_{element}"
            )


if __name__ == "__main__":
    road_path = "dataset/train/roads"
    field_path = "dataset/train/fields"
    creating_file_with_augmented_images(road_path, "dataset/augmented/roads")
    creating_file_with_augmented_images(field_path, "dataset/augmented/fields")

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(crop_size: int = 256, seed: int = 42) -> A.Compose:
    return A.Compose([
        A.RandomCrop(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=40, sigma=5, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0.0, 0.0), per_channel=True, p=0.2),
        A.ShotNoise(scale_range=(0.01, 0.08), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"},
    seed=seed)


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})
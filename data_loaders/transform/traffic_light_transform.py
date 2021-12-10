import os

from base.base_transform import BaseTransform
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np


class TrafficLightTransform(BaseTransform):
    def __call__(self, args):
        image_size = args['image_size']
        image = args['image']
        boxes = args['boxes']
        image_width, image_height = image_size

        augmented_image, augmented_box = self.augment(image, boxes, (image_width, image_height))

        return augmented_image, augmented_box

    def augment(self, image, boxes, image_size):
        image_width, image_height = image_size
        bounding_boxes = []
        for box in boxes:
            augmented_box = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            bounding_boxes.append(augmented_box)
        bbs = BoundingBoxesOnImage(bounding_boxes, shape=image.shape)

        aug_pipeline = iaa.Sequential([
            iaa.Resize((image_width, image_height)),
            iaa.SomeOf((0, 3), [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                iaa.Fliplr(1.0),  # horizontally flip
                iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))),  # crop and pad 50% of the images
                iaa.Sometimes(0.5, iaa.Affine(rotate=5))  # rotate 50% of the images
            ])
        ],
            random_order=True  # apply the augmentations in random order
        )

        image_aug, bbs_aug = aug_pipeline(image=image, bounding_boxes=bbs)
        bbs_aug = np.array([[bb_aug.x1_int, bb_aug.y1_int, bb_aug.x2_int, bb_aug.y2_int] for bb_aug in bbs_aug])
        return image_aug, bbs_aug

# config = parse('../../configs/configs.yaml')
# augmentation = TrafficLightAugmentation(config)
# augmentation()


# config = parse('../../configs/configs.yaml')
# augmentation = TrafficLightAugmentation(config)
# augmentation()

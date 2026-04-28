import math
import random

import cv2
import numpy as np
import torchvision.transforms as T
from matplotlib import pyplot as plt
from utils import get_valid_args, is_dict, is_list

from data import transform as base_transform


class NoOperation:
    def __call__(self, x):
        return x


class BaseSilTransform:
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseCenterTransformPointCloud:
    def __init__(self):
        pass

    def __call__(self, point_cloud):
        # Initialize an empty array to store the centered point clouds
        centered_point_clouds = np.empty_like(point_cloud)

        # Iterate over each point cloud in the sequence
        for i in range(point_cloud.shape[0]):
            # Compute the centroid of the i-th point cloud
            centroid = np.mean(point_cloud[i], axis=0)
            # Center the i-th point cloud
            centered_point_clouds[i] = point_cloud[i] - centroid

        return centered_point_clouds


class BaseCenterAndNormalizeTransformPointCloud:
    def __init__(self):
        pass

    def __call__(self, point_cloud_sequence):
        # Initialize an empty array to store the processed point clouds
        processed_point_clouds = np.empty_like(point_cloud_sequence)

        # Iterate over each point cloud in the sequence
        for i in range(point_cloud_sequence.shape[0]):
            # Center the i-th point cloud
            centroid = np.mean(point_cloud_sequence[i], axis=0)
            centered_point_cloud = point_cloud_sequence[i] - centroid

            # Normalize the i-th point cloud
            max_distance = np.max(np.sqrt(np.sum(centered_point_cloud**2, axis=1)))
            normalized_point_cloud = centered_point_cloud / max_distance

            # Store the processed point cloud
            processed_point_clouds[i] = normalized_point_cloud

        return processed_point_clouds


class BaseParsingCuttingTransform:
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        x = x[..., cutting:-cutting]
        if x.max() == 255 or x.max() == 255.0:
            return x / self.divsor
        else:
            return x / 1.0


class BaseSilCuttingTransform:
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.divsor


class BaseRgbTransform:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std


# **************** Data Agumentation ****************


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1 : x1 + h, y1 : y1 + w] = 0.0
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate([ret], 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            seq = [
                cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)
            ]
            seq = np.concatenate([np.array(_)[np.newaxis, ...] for _ in seq], 0)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        """
        Random perspective transformation applied consistently across the sequence.

        Args:
            prob (float): Probability of applying the transformation
        """
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq

        nf, c, h, w = seq.shape

        # Calculate perspective points once for the whole sequence
        cutting = int(w // 44) * 10
        x_left = list(range(0, cutting))
        x_right = list(range(w - cutting, w))

        # Define source points for perspective transform
        TL = (random.choice(x_left), 0)
        TR = (random.choice(x_right), 0)
        BL = (random.choice(x_left), h)
        BR = (random.choice(x_right), h)

        srcPoints = np.float32([TL, TR, BR, BL])
        canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Get perspective transform matrix once
        perspectiveMatrix = cv2.getPerspectiveTransform(
            np.array(srcPoints), np.array(canvasPoints)
        )

        # Reshape sequence to (nf*c, h, w) to process all frames and channels at once
        reshaped_seq = seq.reshape(-1, h, w)

        # Apply same transformation to all frames and channels
        transformed = np.stack(
            [
                cv2.warpPerspective(frame, perspectiveMatrix, (w, h))
                for frame in reshaped_seq
            ]
        )

        # Reshape back to (nf, c, h, w)
        return transformed.reshape(nf, c, h, w)


class RandomCrop(object):
    def __init__(
        self,
        prob=0.5,
        scale=(0.8, 1.0),
        per_frame=False,
    ):
        """
        Random crop augmentation that maintains original dimensions.

        Args:
            prob (float): Probability of applying the crop
            scale (tuple): Range of size scale to crop. Default crops between 80-100% of original size
        """
        self.prob = prob
        self.scale = scale
        self.per_frame = per_frame

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            if self.per_frame:
                return np.array([self._apply_crop(frame) for frame in seq])
            else:
                return self._apply_crop(seq)

    def _apply_crop(self, seq):
        _, orig_h, orig_w = seq.shape

        # Randomly choose scale factor
        scale_factor = random.uniform(self.scale[0], self.scale[1])

        # Calculate crop dimensions
        crop_h = int(orig_h * scale_factor)
        crop_w = int(orig_w * scale_factor)

        # Calculate valid ranges for top-left corner of crop
        h_start_range = orig_h - crop_h
        w_start_range = orig_w - crop_w

        if h_start_range < 0 or w_start_range < 0:
            return seq

        # Randomly choose top-left corner
        h_start = random.randint(0, h_start_range)
        w_start = random.randint(0, w_start_range)

        # Crop and resize each frame
        # Crop frame
        cropped = seq[:, h_start : h_start + crop_h, w_start : w_start + crop_w]
        # Resize back to original dimensions
        resized = cv2.resize(cropped.transpose(1, 2, 0), (orig_w, orig_h)).transpose(
            2, 0, 1
        )

        return resized


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32(
                [
                    [random.choice(shift_range), random.choice(shift_range)],
                    [dh - random.choice(shift_range), random.choice(shift_range)],
                    [random.choice(shift_range), dw - random.choice(shift_range)],
                ]
            )
            pts2 = np.float32(
                [
                    [random.choice(shift_range), random.choice(shift_range)],
                    [dh - random.choice(shift_range), random.choice(shift_range)],
                    [random.choice(shift_range), dw - random.choice(shift_range)],
                ]
            )
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            seq = [
                cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)
            ]
            seq = np.concatenate([np.array(_)[np.newaxis, ...] for _ in seq], 0)
            return seq


class LGT(object):
    def __init__(self, prob=0.2, sl=0.02, sh=0.4, r1=0.3, per_frame=False):
        """
        Local Grayscale Transformation for sequences.

        Args:
            prob (float): Probability of applying the transformation
            sl (float): Minimum area ratio
            sh (float): Maximum area ratio
            r1 (float): Aspect ratio range
            per_frame (bool): Whether to apply different transformations per frame
        """
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def transform_single(self, seq_frame):
        """Apply LGT to a single frame."""
        if random.uniform(0, 1) >= self.prob:
            return seq_frame

        # Get frame dimensions
        h, w, _ = seq_frame.shape
        area = h * w

        # Convert to grayscale
        gray_frame = cv2.cvtColor(seq_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        for _ in range(100):
            # Calculate random region size
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            patch_h = int(round(math.sqrt(target_area * aspect_ratio)))
            patch_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if patch_w < w and patch_h < h:
                # Random position
                x1 = random.randint(0, h - patch_h)
                y1 = random.randint(0, w - patch_w)

                # Replace region with grayscale
                seq_frame[x1 : x1 + patch_h, y1 : y1 + patch_w] = gray_frame[
                    x1 : x1 + patch_h, y1 : y1 + patch_w, np.newaxis
                ]

                return seq_frame

        return seq_frame

    def __call__(self, seq):
        """
        Apply LGT to sequence.

        Args:
            seq: Input sequence of shape (frames, height, width)
        Returns:
            Transformed sequence of same shape
        """
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq

            # Apply same transformation to all frames
            h, w = seq.shape[2:]
            area = h * w

            # Calculate random region size
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            patch_h = int(round(math.sqrt(target_area * aspect_ratio)))
            patch_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if patch_w < w and patch_h < h:
                # Random position (same for all frames)
                x1 = random.randint(0, h - patch_h)
                y1 = random.randint(0, w - patch_w)

                # Transform each frame
                transformed_frames = []
                for frame in np.split(seq, seq.shape[0], axis=0):
                    frame = frame[0]  # Remove channel dimension
                    gray_frame = cv2.cvtColor(
                        frame.astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY
                    )
                    frame[:, x1 : x1 + patch_h, y1 : y1 + patch_w] = gray_frame[
                        x1 : x1 + patch_h, y1 : y1 + patch_w
                    ][None, ...].repeat(3, 0)
                    transformed_frames.append(frame)

                return np.concatenate([transformed_frames], 0)

            return seq
        else:
            # Apply different transformation to each frame
            transformed_frames = []
            for frame in np.split(seq, seq.shape[0], axis=0):
                transformed = self.transform_single(
                    frame[0].transpose(1, 2, 0)  # (H, W, C)
                ).transpose(2, 0, 1)  # Remove channel dimension
                transformed_frames.append(transformed[np.newaxis, ...])

            return np.concatenate(transformed_frames, 0)


class ColorJitter(object):
    def __init__(self, prob=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        Color jittering augmentation.

        Args:
            prob (float): Probability of applying the jitter
            brightness (float): How much to jitter brightness (0-1)
            contrast (float): How much to jitter contrast (0-1)
            saturation (float): How much to jitter saturation (0-1)
            hue (float): How much to jitter hue (0-0.5)
        """
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = min(hue, 0.5)  # Hue rotation should be between -0.5 and 0.5

    def adjust_brightness(self, frame, factor):
        """Adjust brightness of an image by multiplying with factor."""
        return np.clip(frame * factor, 0, 255)

    def adjust_contrast(self, frame, factor):
        """Adjust contrast of an image."""
        mean = np.mean(frame, axis=(0, 1))
        return np.clip(mean + factor * (frame - mean), 0, 255)

    def adjust_saturation(self, frame, factor):
        """Adjust saturation of an image."""
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.clip(gray + factor * (frame - gray), 0, 255)

    def adjust_hue(self, frame, factor):
        """Adjust hue of an image."""
        hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq

        # Create random factors for each adjustment
        brightness_factor = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness
        )
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation
        )
        hue_factor = random.uniform(-self.hue, self.hue)

        # Randomly shuffle the order of adjustments
        transforms = [
            (self.adjust_brightness, brightness_factor),
            (self.adjust_contrast, contrast_factor),
            (self.adjust_saturation, saturation_factor),
            (self.adjust_hue, hue_factor),
        ]
        random.shuffle(transforms)

        # Apply transforms frame by frame
        jittered_frames = []
        for frame in np.split(seq, seq.shape[0], axis=0):
            img = frame[0]  # Remove the channel dimension for processing

            # Apply each transform in random order
            for transform_func, factor in transforms:
                img = transform_func(img, factor)

            jittered_frames.append(img[np.newaxis, ...])

        # Concatenate frames back together
        seq = np.concatenate(jittered_frames, 0)

        return seq.astype(np.float32)


class RandomGaussianNoiseAccordingToColorMap(object):
    def __init__(self, prob=0.5, per_frame=False, mean=0.0, std=0.1):
        """
        Args:
            prob (float): The probability of applying the augmentation.
            per_frame (bool): If True, apply noise to each frame separately.
            mean (float): Mean for the Gaussian noise.
            std (float): Standard deviation for the Gaussian noise.
        """
        self.prob = prob
        self.per_frame = per_frame
        self.mean = mean
        self.std = std
        self.colormap = plt.get_cmap("jet")

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            if self.per_frame:
                # Apply Gaussian noise per frame separately
                return np.array([self._apply_noise(frame) for frame in seq])
            else:
                # Apply Gaussian noise to the entire sequence
                return self._apply_noise(seq)

    def _apply_noise(self, seq):
        # Convert to float32 for precise noise addition
        seq = seq.astype(np.float32)

        # Check if the sequence is black (all zeros)
        if np.all(seq == 0):
            return seq

        # Generate Gaussian noise
        noise = np.random.normal(self.mean, self.std, seq.shape[1:])

        # Normalize the noise according to the jet colormap
        normed_noise = self.colormap(
            (noise - noise.min()) / (noise.max() - noise.min())
        )[..., :3]  # Get RGB values
        normed_noise = normed_noise.transpose(2, 0, 1)  # (C, H, W)

        # Apply the noise
        # Apply the noise to 40% of the image, except for black pixels
        mask = (seq != 0).all(0).astype(np.float32)
        noise_mask = np.random.rand(*seq.shape[1:]) < 0.4
        combined_mask = mask * noise_mask
        seq = seq * (1 - combined_mask[None]) + normed_noise * combined_mask[None]

        return seq


# ******************************************


def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if trf_cfg is None:
        return lambda x: x
    elif is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg["type"])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ["type"])
        return transform(**valid_trf_arg)
    elif is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    else:
        raise ValueError("Error type for -Transform-Cfg-")


# **************** For pose ****************
class RandomSelectSequence(object):
    """
    Randomly select different subsequences
    """

    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            raise ValueError(
                "The sequence length of data is too short, which does not meet the requirements."
            )
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    """
    Select center subsequence
    """

    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0] / 2) - (self.sequence_length / 2))
        except ValueError:
            raise ValueError(
                "The sequence length of data is too short, which does not meet the requirements."
            )
        end = start + self.sequence_length
        return data[start:end]


class MirrorPoses(object):
    """
    Performing Mirror Operations
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if np.random.random() <= self.prob:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class NormalizeEmpty(object):
    """
    Normliza Empty Joint
    """

    def __call__(self, data):
        frames, joints = np.where(data[:, :, 0] == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = np.mean(data[frame], axis=0)
            data[frame, joint, 0] = center_of_gravity[0]
            data[frame, joint, 1] = center_of_gravity[1]
            data[frame, joint, 2] = 0
        return data


class RandomMove(object):
    """
    Move: add Random Movement to each joint
    """

    def __init__(self, random_r=[4, 1]):
        self.random_r = random_r

    def __call__(self, data):
        noise = np.zeros(3)
        noise[0] = np.random.uniform(-self.random_r[0], self.random_r[0])
        noise[1] = np.random.uniform(-self.random_r[1], self.random_r[1])
        data += np.tile(noise, (data.shape[0], data.shape[1], 1))
        return data


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class FlipSequence(object):
    """
    Temporal Fliping
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0).copy()
        return data


class InversePosesPre(object):
    """
    Left-right flip of skeletons
    """

    def __init__(self, probability=0.5, joint_format="coco"):
        self.probability = probability
        if joint_format == "coco":
            self.invers_arr = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        elif joint_format in ["alphapose", "openpose"]:
            self.invers_arr = [
                0,
                1,
                5,
                6,
                7,
                2,
                3,
                4,
                11,
                12,
                13,
                8,
                9,
                10,
                15,
                14,
                17,
                16,
            ]
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, data):
        for i in range(len(data)):
            if np.random.random() <= self.probability:
                data[i] = data[i, self.invers_arr, :]
        return data


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.25):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack(
            (
                np.random.normal(0, self.std, (data.shape[1], 2)),
                np.zeros((data.shape[1], 1)),
            )
        ).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class TwoView(object):
    def __init__(self, trf_cfg):
        assert is_list(trf_cfg)
        self.transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])

    def __call__(self, data):
        return np.concatenate([self.transform(data), self.transform(data)], axis=1)


class MSGGTransform:
    def __init__(self, joint_format="coco"):
        if joint_format == "coco":  # 17
            self.mask = [6, 8, 14, 12, 7, 13, 5, 10, 16, 11, 9, 15]
        elif joint_format in ["alphapose", "openpose"]:  # 18
            self.mask = [2, 3, 9, 8, 6, 12, 5, 4, 10, 11, 7, 13]
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, x):
        result = x[..., self.mask, :].copy()
        return result

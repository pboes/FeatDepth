import random
from .mono_dataset import MonoDataset
import os
import PIL.Image as pil  # using pillow-simd for increased speed

import torch
from torchvision import transforms


class TrafficCamsDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(TrafficCamsDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        if not self.is_train and self.gt_depth_path is not None:
            gt_depth = self.gt_depths[index]
            inputs["gt_depth"] = gt_depth

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(index, i, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
        return inputs

    def get_color(self, frame_index, relative_index, do_flip):
        if (frame_index == 0) and (relative_index <= 0):
            other_index = 0
        elif (frame_index == len(self.filenames)) and (relative_index >= 0):
            other_index = len(self.filenames)
        else:
            other_index = frame_index + relative_index

        color = self.loader(os.path.join(self.data_path, self.filenames[other_index]))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

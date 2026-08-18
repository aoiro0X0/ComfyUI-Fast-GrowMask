import unittest

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from utility.fast_mask import expand_mask_batch, gaussian_blur_like_pillow


class FastMaskTests(unittest.TestCase):
    def test_square_dilation_matches_repeated_three_by_three(self):
        torch.manual_seed(7)
        mask = (torch.rand(4, 96, 96) > 0.72).float()

        actual = expand_mask_batch(mask, [12] * 4, tapered_corners=False)
        expected = mask.unsqueeze(1)
        for _ in range(12):
            expected = F.max_pool2d(expected, 3, stride=1, padding=1)

        self.assertTrue(torch.equal(actual, expected.squeeze(1)))

    def test_square_erosion_matches_repeated_three_by_three(self):
        torch.manual_seed(9)
        mask = (torch.rand(4, 96, 96) > 0.28).float()

        actual = expand_mask_batch(mask, [-7] * 4, tapered_corners=False)
        expected = -mask.unsqueeze(1)
        for _ in range(7):
            expected = F.max_pool2d(expected, 3, stride=1, padding=1)

        self.assertTrue(torch.equal(actual, -expected.squeeze(1)))

    def test_per_frame_expand_amounts_preserve_batch_order(self):
        mask = torch.zeros(4, 32, 32)
        mask[:, 16, 16] = 1
        amounts = [4, 0, 2, 4]

        actual = expand_mask_batch(mask, amounts, tapered_corners=False)
        expected = []
        for frame, amount in zip(mask, amounts):
            work = frame[None, None]
            for _ in range(amount):
                work = F.max_pool2d(work, 3, stride=1, padding=1)
            expected.append(work.squeeze())

        self.assertTrue(torch.equal(actual, torch.stack(expected)))

    def test_blur_matches_original_pillow_path(self):
        sample = torch.zeros(2, 192, 192)
        sample[:, 64:128, 72:120] = 1
        sample[:, 88:104, 48:144] = 1

        for radius in (3, 16, 64, 100):
            with self.subTest(radius=radius):
                actual = gaussian_blur_like_pillow(sample, radius)
                expected_frames = []
                for frame in sample:
                    image = Image.fromarray(
                        np.clip(frame.numpy() * 255, 0, 255).astype(np.uint8)
                    )
                    blurred = image.filter(ImageFilter.GaussianBlur(radius))
                    expected_frames.append(
                        torch.from_numpy(np.asarray(blurred).copy()).float() / 255
                    )
                expected = torch.stack(expected_frames)
                self.assertTrue(torch.equal(actual, expected))

    def test_blur_matches_original_pillow_path_for_soft_masks(self):
        torch.manual_seed(23)
        sample = torch.rand(2, 127, 193)

        for radius in (0.1, 7.5, 32, 64):
            with self.subTest(radius=radius):
                actual = gaussian_blur_like_pillow(sample, radius)
                expected_frames = []
                for frame in sample:
                    image = Image.fromarray(
                        np.clip(frame.numpy() * 255, 0, 255).astype(np.uint8)
                    )
                    blurred = image.filter(ImageFilter.GaussianBlur(radius))
                    expected_frames.append(
                        torch.from_numpy(np.asarray(blurred).copy()).float() / 255
                    )
                expected = torch.stack(expected_frames)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

import ast
from pathlib import Path
import unittest

import torch


def load_split_method():
    source_path = Path(__file__).parents[1] / "nodes" / "image_nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    split_method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SplitImageChannels"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "split"
    )
    module = ast.Module(body=[split_method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["split"]


class SplitImageChannelsTests(unittest.TestCase):
    def test_rgba_uses_fourth_channel_as_alpha_mask(self):
        split = load_split_method()
        image = torch.rand(2, 17, 23, 4)

        red, green, blue, alpha = split(None, image)

        self.assertTrue(torch.equal(red, image[..., 0:1].expand(-1, -1, -1, 3)))
        self.assertTrue(torch.equal(green, image[..., 1:2].expand(-1, -1, -1, 3)))
        self.assertTrue(torch.equal(blue, image[..., 2:3].expand(-1, -1, -1, 3)))
        self.assertTrue(torch.equal(alpha, image[..., 3]))

    def test_rgb_returns_zero_alpha_mask(self):
        split = load_split_method()
        image = torch.rand(2, 17, 23, 3)

        _, _, _, alpha = split(None, image)

        self.assertEqual(alpha.shape, image.shape[:3])
        self.assertEqual(alpha.device, image.device)
        self.assertTrue(torch.equal(alpha, torch.zeros_like(alpha)))


if __name__ == "__main__":
    unittest.main()

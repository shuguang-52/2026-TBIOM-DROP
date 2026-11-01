"""Real CPU model tests; no datasets, pretrained downloads, or CUDA required.

Run with: python -m unittest discover -s tests -p test_model_smoke.py -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from torchreid.models import build_model
from torchreid.utils.constants import GLOBAL, PARTS


def model_config(dim_reduce='after_pooling', pooling='gwamp'):
    return SimpleNamespace(model=SimpleNamespace(bpbreid=SimpleNamespace(
        backbone='hrnet32', use_decouple='fpn', last_stride=1,
        dim_reduce=dim_reduce, dim_reduce_output=512,
        hrnet_pretrained_path='', masks=SimpleNamespace(parts_num=8),
        shared_parts_id_classifier=False, test_use_target_segmentation='none',
        training_binary_visibility_score=True, testing_binary_visibility_score=True,
        normalization='identity', pooling=pooling, learnable_attention_enabled=True,
    )))


class ModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_eval_at_both_supported_image_heights(self):
        model = build_model('drop', 3, pretrained=False, config=model_config()).eval()
        for height, feature_height in ((256, 64), (384, 96)):
            with self.subTest(height=height), torch.inference_mode():
                outputs = model(torch.randn(1, 3, height, 128))
                embeddings, visibility, scores, pixels, spatial, masks = outputs
                self.assertEqual(tuple(embeddings[GLOBAL].shape), (1, 512))
                self.assertEqual(tuple(embeddings[PARTS].shape), (1, 8, 512))
                self.assertEqual(tuple(visibility[PARTS].shape), (1, 8))
                self.assertEqual(tuple(scores[PARTS].shape), (1, 8, 3))
                self.assertEqual(tuple(pixels.shape), (1, 9, feature_height, 32))
                self.assertEqual(tuple(spatial.shape), (1, 1920, feature_height, 32))
                self.assertEqual(tuple(masks[PARTS].shape), (1, 8, feature_height, 32))
                for value in (*embeddings.values(), *scores.values(), pixels, spatial):
                    self.assertTrue(torch.isfinite(value).all().item())
                    self.assertEqual(value.device.type, 'cpu')

    def test_before_pooling_reduction_remains_usable(self):
        config = model_config(dim_reduce='before_pooling', pooling='gwap')
        model = build_model('drop', 3, pretrained=False, config=config).eval()
        with torch.inference_mode():
            embeddings, _, _, _, spatial, _ = model(torch.randn(1, 3, 256, 128))
        self.assertEqual(tuple(spatial.shape), (1, 512, 64, 32))
        self.assertEqual(tuple(embeddings[PARTS].shape), (1, 8, 512))
        self.assertTrue(torch.isfinite(embeddings[PARTS]).all().item())

    def test_training_backward_reaches_fpn_and_position_encoding(self):
        model = build_model('bpbreid', 3, pretrained=False, config=model_config()).train()
        _, _, scores, pixels, _, _ = model(torch.randn(2, 3, 256, 128))
        loss = F.cross_entropy(scores[GLOBAL], torch.tensor([0, 1]))
        loss = loss + F.cross_entropy(pixels, torch.randint(9, (2, 64, 32)))
        loss.backward()
        backbone = model.backbone_appearance_feature_extractor
        for name, parameter in (
            ('stem', backbone.conv1.weight),
            ('fpn', backbone.FPN.TSDecs1[0].conv.weight),
            ('projection', model.reduce_dim[0][0].weight),
            ('position', model.pos_embeding[0].weight),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # The original task prompt is zero-initialized. Its BN/ReLU branch can
        # have zero gradient initially; it must still participate in autograd.
        task_gradient = backbone.FPN.TaskPro1.grad
        self.assertIsNotNone(task_gradient)
        self.assertTrue(torch.isfinite(task_gradient).all().item())

    def test_pretrained_backbone_weights_are_loaded_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hrnetv2_w32_imagenet_pretrained.pth'
            expected = torch.full((64, 3, 3, 3), 0.125)
            torch.save({'conv1.weight': expected}, path)
            model = build_model('hrnet32', 3, pretrained=True, pretrained_path=directory)
            torch.testing.assert_close(model.conv1.weight, expected)


if __name__ == '__main__':
    unittest.main()

"""CPU training loss and image-only data regressions using synthetic samples."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts_reid.default_config import get_default_config, imagedata_kwargs
from torchreid.data import ImageDataManager
from torchreid.data.masks_transforms import compute_parts_num_and_names
from torchreid.engine import ImagePartBasedEngine
from torchreid.losses.GiLt_loss import GiLtLoss
from torchreid.losses.body_part_attention_loss import BodyPartAttentionLoss
from torchreid.utils.constants import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS


class TrainingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gilt_and_pixel_losses_backpropagate_on_cpu_with_memory(self):
        weights = copy.deepcopy(GiLtLoss.default_losses_weights)
        weights[FOREGROUND]['tr'] = 1.0
        # Exercise the real engine loss combination, including target-mask resizing.
        engine = ImagePartBasedEngine.__new__(ImagePartBasedEngine)
        engine.GiLt = GiLtLoss(weights, part_num=8, bank_size=16, use_gpu=False)
        engine.body_part_attention_loss = BodyPartAttentionLoss(with_ss=True, use_gpu=False)
        pids = torch.tensor([0, 0, 1, 1])
        for _ in range(2):
            embeddings = {
                GLOBAL: torch.randn(4, 512, requires_grad=True),
                FOREGROUND: torch.randn(4, 512, requires_grad=True),
                CONCAT_PARTS: torch.randn(4, 4096, requires_grad=True),
                PARTS: torch.randn(4, 8, 512, requires_grad=True),
            }
            scores = {key: torch.randn(4, 3, requires_grad=True)
                      for key in (GLOBAL, FOREGROUND, CONCAT_PARTS)}
            visibility = {key: torch.ones(4) for key in scores}
            visibility[PARTS] = torch.ones(4, 8)
            pixels = torch.randn(4, 9, 8, 4, requires_grad=True)
            masks = torch.rand(4, 9, 16, 8)
            loss, summary = engine.combine_losses(
                visibility, embeddings, scores, pids, pixels, masks, bpa_weight=0.4,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertIn(PARTS, summary)
            for tensor in (embeddings[FOREGROUND], embeddings[PARTS], pixels, *scores.values()):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertEqual(engine.GiLt.part_triplet_loss.parts_bank.bank.device.type, 'cpu')
        self.assertFalse(engine.GiLt.part_triplet_loss.parts_bank.bank.requires_grad)

    def test_spatial_smoothing_matches_released_formula(self):
        pixels = torch.randn(2, 9, 8, 4, requires_grad=True)
        labels = torch.randint(9, (2, 8, 4))
        criterion = BodyPartAttentionLoss(with_ss=True, use_gpu=False)
        loss, _ = criterion(pixels, labels)
        smoothness = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean()
        smoothness = smoothness + (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean()
        expected = F.cross_entropy(pixels, labels, label_smoothing=0.1) + 0.35 * smoothness
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(pixels.grad).all().item())

    def test_image_loader_uses_train_masks_but_no_query_or_gallery_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'Market-1501-v15.09.15'
            for split in ('bounding_box_train', 'query', 'bounding_box_test'):
                (dataset / split).mkdir(parents=True)
                for pid in (1, 2):
                    cameras = (1, 2) if split == 'bounding_box_train' else (1,)
                    for camera in cameras:
                        stem = f'{pid:04d}_c{camera}_0001'
                        Image.fromarray(np.full((64, 32, 3), 64 * pid, dtype=np.uint8)).save(
                            dataset / split / (stem + '.jpg')
                        )
                        if split == 'bounding_box_train':
                            folder = dataset / 'masks' / 'pifpaf_maskrcnn_filtering' / split
                            folder.mkdir(parents=True, exist_ok=True)
                            np.save(folder / (stem + '.npy'), np.ones((36, 16, 8), dtype=np.float32))
            cfg = get_default_config()
            cfg.use_gpu = False
            cfg.data.root = directory
            cfg.data.height, cfg.data.width = 64, 32
            cfg.data.workers = 0
            cfg.data.transforms = []
            cfg.train.batch_size = 4
            cfg.test.batch_size = 2
            cfg.sampler.num_instances = 2
            compute_parts_num_and_names(cfg)
            manager = ImageDataManager(**imagedata_kwargs(cfg))
            train = next(iter(manager.train_loader))
            self.assertEqual(tuple(train['mask'].shape), (4, 9, 16, 8))
            self.assertEqual(tuple(train['image'].shape), (4, 3, 64, 32))
            for split in ('query', 'gallery'):
                batch = next(iter(manager.test_loader['market1501'][split]))
                self.assertNotIn('mask', batch)
                self.assertEqual(tuple(batch['image'].shape), (2, 3, 64, 32))
            # Cached train metadata must retain its own mask policy.
            self.assertIn('mask', next(iter(manager.train_loader)))
            self.assertNotIn('mask', manager.train_loader.dataset.train[0])


if __name__ == '__main__':
    unittest.main()

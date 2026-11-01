"""Public entry points and CPU checkpoint loading regressions."""
import tempfile
import unittest
from pathlib import Path
import torch
from scripts_reid.default_config import get_default_config
from scripts_reid.main import validate_config
from torchreid.data.masks_transforms import compute_parts_num_and_names
from torchreid.utils import load_checkpoint, load_pretrained_weights

ROOT = Path(__file__).resolve().parents[1]

class ReleaseRuntimeTests(unittest.TestCase):
    def test_all_release_configs_merge_and_resolve_parts(self):
        paths = sorted((ROOT / "configs/drop").glob("*.yaml"))
        self.assertEqual(len(paths), 11)
        for path in paths:
            with self.subTest(config=path.name):
                cfg = get_default_config()
                cfg.merge_from_file(str(path))
                compute_parts_num_and_names(cfg)
                self.assertGreater(cfg.model.bpbreid.masks.parts_num, 0)
                if cfg.test.evaluate:
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        validate_config(cfg)
                else:
                    validate_config(cfg)

    def test_checkpoint_loads_on_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            torch.save({"state_dict": {"weight": torch.ones(2, 3)}}, path)
            checkpoint = load_checkpoint(str(path))
            self.assertTrue(torch.equal(checkpoint["state_dict"]["weight"].cpu(), torch.ones(2, 3)))

    def test_unmatched_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            torch.save({"state_dict": {"unrelated": torch.ones(1)}}, path)
            with self.assertRaisesRegex(ValueError, "No matching"):
                load_pretrained_weights(torch.nn.Linear(3, 2), str(path))

    def test_evaluation_rejects_missing_checkpoint(self):
        cfg = get_default_config()
        cfg.test.evaluate = True
        cfg.model.load_weights = "missing-drop-checkpoint.pth"
        with self.assertRaises(FileNotFoundError):
            validate_config(cfg)

if __name__ == "__main__":
    unittest.main()

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STAGE2E = ROOT / "src" / "stage2e"


class SourceContracts(unittest.TestCase):
    def test_every_persistent_buffer_is_checkpointed(self):
        header = (STAGE2E / "memory_allocator.cuh").read_text(encoding="utf-8")
        body = header.split("struct PersistentBuffers {", 1)[1].split("};", 1)[0]
        fields = set(re.findall(r"\b(d_[A-Za-z0-9_]+)\s*;", body))
        checkpoint = (STAGE2E / "scheduler_checkpoint.cu").read_text(encoding="utf-8")
        missing = sorted(name for name in fields if f"b.{name}" not in checkpoint)
        self.assertEqual([], missing)

    def test_every_scheduler_device_buffer_is_checkpointed(self):
        header = (STAGE2E / "scheduler.cuh").read_text(encoding="utf-8")
        fields = set(re.findall(r"\b(d_[A-Za-z0-9_]+)_\s*;", header))
        checkpoint = (STAGE2E / "scheduler_checkpoint.cu").read_text(encoding="utf-8")
        missing = sorted(name for name in fields if f"self->{name}_" not in checkpoint)
        self.assertEqual([], missing)

    def test_linux_sources_do_not_use_windows_directory_api(self):
        scheduler = (STAGE2E / "scheduler.cu").read_text(encoding="utf-8")
        self.assertNotIn("<direct.h>", scheduler)
        self.assertNotIn("_mkdir(", scheduler)

    def test_stage2e_cli_documents_resume_contract(self):
        config = (STAGE2E / "run_config.cpp").read_text(encoding="utf-8")
        for option in (
            "--steps", "--device", "--seed", "--text", "--checkpoint-dir",
            "--checkpoint-interval", "--keep-checkpoints", "--resume",
        ):
            self.assertIn(option, config)

    def test_training_script_runs_from_project_root(self):
        script = (STAGE2E / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn('cd "$PROJECT_ROOT"', script)
        self.assertIn("data/lccc_sample_1mb.txt", script)


if __name__ == "__main__":
    unittest.main()

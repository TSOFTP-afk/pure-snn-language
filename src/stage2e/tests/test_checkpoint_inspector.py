import importlib.util
import tempfile
import unittest
import zlib
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "inspect_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_checkpoint", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def checksum(data: bytes) -> int:
    return zlib.crc32(data)


class CheckpointInspectorTests(unittest.TestCase):
    def make_checkpoint(self, path: Path, payload: bytes):
        digest = checksum(payload)
        header = MODULE.HEADER.pack(
            MODULE.MAGIC, 3, MODULE.HEADER.size, 1, 0, len(payload), digest,
            55_000, 10_700_000, 80, 56,
        )
        section = MODULE.SECTION.pack(b"scheduler_state", len(payload))
        footer = MODULE.FOOTER.pack(MODULE.FOOTER_MAGIC, digest)
        path.write_bytes(header + section + payload + footer)

    def test_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.snn2e"
            self.make_checkpoint(path, b"checkpoint-payload")
            result = MODULE.inspect(path, verify=True)
            self.assertTrue(result["checksum_verified"])
            self.assertEqual(result["n_neurons"], 55_000)

    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.snn2e"
            self.make_checkpoint(path, b"checkpoint-payload")
            raw = bytearray(path.read_bytes())
            raw[MODULE.HEADER.size + MODULE.SECTION.size] ^= 0x01
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum"):
                MODULE.inspect(path, verify=True)


if __name__ == "__main__":
    unittest.main()

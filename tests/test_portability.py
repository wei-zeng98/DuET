"""Cross-platform file handling regression tests."""

import tempfile
import unittest
from pathlib import Path

from duet.pipeline import _sha256


class PortabilityTests(unittest.TestCase):
    def test_vocabulary_hash_is_independent_of_checkout_newlines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(b'{\n  "value": 1\n}\n')
            crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')

            self.assertEqual(
                _sha256(lf, normalize_newlines=True),
                _sha256(crlf, normalize_newlines=True),
            )
            self.assertNotEqual(_sha256(lf), _sha256(crlf))


if __name__ == "__main__":
    unittest.main()

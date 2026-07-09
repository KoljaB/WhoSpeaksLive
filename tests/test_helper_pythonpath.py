from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.pythonpath import build_pythonpath


class HelperPythonPathTests(unittest.TestCase):
    def test_build_pythonpath_skips_site_packages_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site_packages = root / "lib" / "python3.12" / "site-packages"
            normal_path = root / "extra-src"
            site_packages.mkdir(parents=True)
            normal_path.mkdir()

            result = build_pythonpath((site_packages,), os.pathsep.join([str(site_packages), str(normal_path)]))

            parts = result.split(os.pathsep)
            self.assertNotIn(str(site_packages.resolve()), parts)
            self.assertIn(str(normal_path.resolve()), parts)


if __name__ == "__main__":
    unittest.main()

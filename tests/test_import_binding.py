import os
import unittest

import Cozter


class ImportBindingTests(unittest.TestCase):
    def test_imports_use_this_checkout(self) -> None:
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        imported = os.path.realpath(os.path.dirname(Cozter.__file__ or ""))

        self.assertEqual(imported, os.path.realpath(package_dir))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for astronomical intensity handling, without loading Torch."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import load_input, save_output


class InferenceIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_both_tiff_byte_orders_preserve_normalized_intensity(self):
        for dtype in ('<u2', '>u2'):
            path = self.root / ('little.tif' if dtype[0] == '<' else 'big.tif')
            Image.fromarray(np.full((256, 256), 32768, dtype=dtype)).save(path)
            np.testing.assert_allclose(load_input(path), 32768 / 65535, rtol=1e-6)

    def test_sixteen_bit_png(self):
        path = self.root / 'gray16.png'
        Image.fromarray(np.full((256, 256), 65535, dtype=np.uint16)).save(path)
        np.testing.assert_array_equal(load_input(path), np.ones((256, 256)))

    def test_rgb_requires_explicit_grayscale_conversion(self):
        path = self.root / 'rgb.png'
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'grayscale plane'):
            load_input(path)

    def test_nonfinite_input_rejected(self):
        path = self.root / 'bad.npy'
        array = np.zeros((256, 256), dtype=np.float32)
        array[10, 10] = np.nan
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, 'NaN'):
            load_input(path)

    def test_npy_intensity_and_center_crop(self):
        path = self.root / 'image.npy'
        array = np.arange(258 * 258, dtype=np.float32).reshape(258, 258)
        np.save(path, array)
        np.testing.assert_array_equal(load_input(path), array[1:-1, 1:-1] / 2500)

    def test_fts_output_round_trip(self):
        path = self.root / 'restored.fts'
        plane = np.full((256, 256), 0.5, dtype=np.float32)
        save_output(path, plane, scale_back=True)
        np.testing.assert_array_equal(load_input(path), plane)


if __name__ == '__main__':
    unittest.main()

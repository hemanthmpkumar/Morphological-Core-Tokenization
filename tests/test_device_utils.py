import os
import unittest
import torch

from src.utils.device import get_compute_device


class TestDeviceUtils(unittest.TestCase):
    """Ensure default device selection behaves as expected."""

    def setUp(self):
        # backup original functions so we can restore later
        self._orig_cuda = torch.cuda.is_available
        self._orig_mps = torch.backends.mps.is_available
        # clear env vars that may affect behavior
        self._orig_env = os.environ.get("MCT_DEVICE")
        if "MCT_DEVICE" in os.environ:
            del os.environ["MCT_DEVICE"]

    def tearDown(self):
        torch.cuda.is_available = self._orig_cuda
        torch.backends.mps.is_available = self._orig_mps
        if self._orig_env is not None:
            os.environ["MCT_DEVICE"] = self._orig_env
        elif "MCT_DEVICE" in os.environ:
            del os.environ["MCT_DEVICE"]

    def test_prefers_cuda_when_available(self):
        torch.cuda.is_available = lambda: True
        torch.backends.mps.is_available = lambda: True
        device = get_compute_device()
        self.assertEqual(str(device), "cuda")

    def test_falls_back_to_cpu_if_no_cuda_and_no_mps(self):
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: False
        device = get_compute_device()
        self.assertEqual(str(device), "cpu")

    def test_does_not_use_mps_by_default(self):
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: True
        device = get_compute_device()
        self.assertEqual(str(device), "cpu")

    def test_allows_mps_if_requested(self):
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: True
        device = get_compute_device(allow_mps=True)
        self.assertEqual(str(device), "mps")

    def test_environment_override(self):
        os.environ["MCT_DEVICE"] = "cpu"
        device = get_compute_device()
        self.assertEqual(str(device), "cpu")

        os.environ["MCT_DEVICE"] = "cuda:0"
        device = get_compute_device()
        self.assertEqual(str(device), "cuda:0")

        os.environ["MCT_DEVICE"] = "mps"
        # override should win even though allow_mps is False
        device = get_compute_device()
        self.assertEqual(str(device), "mps")

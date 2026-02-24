import os
import torch


def get_compute_device(prefer_cuda: bool = True, allow_mps: bool = False) -> torch.device:
    """
    Determine the best device for computation.

    Priority order:
    1. Environment variable override.  The following variables are checked in
       order: ``MCT_DEVICE`` then ``TORCH_DEVICE``.  If either is set to a
       valid device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"`` or
       ``"mps"``) that device is returned immediately.
    2. CUDA (``"cuda"``) when ``prefer_cuda`` is True and a CUDA device is
       available.  This ensures that systems with NVIDIA/AMD GPUs default to
       the fastest backend.
    3. Apple MPS (``"mps"``) *only* when ``allow_mps`` is True and no CUDA
       device is present.  We deliberately avoid using MPS by default because
       most experimental runs target GPU clusters; falling back to CPU is
       preferable if CUDA is not present.
    4. CPU (``"cpu"``) as the final fallback.

    Args:
        prefer_cuda: Whether to prefer a CUDA device if available.
        allow_mps:  Whether MPS should be considered when CUDA is not
                    available.  Defaults to False to avoid accidental Apple
                    MPS usage.

    Returns:
        A ``torch.device`` object pointing to the chosen backend.
    """

    # 1. environment override
    for var in ("MCT_DEVICE", "TORCH_DEVICE"):
        val = os.environ.get(var)
        if val:
            try:
                return torch.device(val)
            except Exception:
                # ignore invalid values and continue looking
                pass

    # 2. CUDA
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    # 3. optional MPS
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    # 4. fallback to CPU
    return torch.device("cpu")

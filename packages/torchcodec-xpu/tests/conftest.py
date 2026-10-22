# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import subprocess

import pytest

# Frames in the generated clip: 4 seconds at 30 fps.
CLIP_NUM_FRAMES = 120


@pytest.fixture(scope="session")
def xpu_clip(tmp_path_factory) -> str:
    """Path to a small H.264 clip, decodable on XPU.

    Skips the test if there is no XPU device or no ffmpeg to generate the clip.
    """
    import torch

    if not torch.xpu.is_available():
        pytest.skip("XPU device is not available")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to generate the test clip")

    path = tmp_path_factory.mktemp("clip") / "testsrc.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=30",
            "-frames:v", str(CLIP_NUM_FRAMES),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return str(path)

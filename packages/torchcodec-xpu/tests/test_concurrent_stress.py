# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stress test for multi-threaded decoding on XPU.

Several threads decode random frame ranges with their own decoders at the same
time. Failures are crashes, aborts and GPU resets, which cannot be caught
in-process, so the workload runs in a subprocess with a timeout. The subprocess
sets USE_SYCL_KERNELS=0 explicitly (the VAAPI path).
"""

import os
import subprocess
import sys

NUM_THREADS = 8
ITERATIONS = 30
TIMEOUT_SECONDS = 120

WORKLOAD = """
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from torchcodec.decoders import VideoDecoder

path, num_threads, iterations = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def work(seed):
    rng = random.Random(seed)
    decoder = VideoDecoder(path, device="xpu", seek_mode="approximate")
    for _ in range(iterations):
        start = rng.randrange(0, 90)
        stop = start + rng.randrange(1, 30)
        decoder.get_frames_in_range(start, stop).data.cpu()


with ThreadPoolExecutor(num_threads) as pool:
    for future in [pool.submit(work, seed) for seed in range(num_threads)]:
        future.result()
torch.xpu.synchronize()
"""


def test_concurrent_random_range_decode_does_not_crash(xpu_clip: str) -> None:
    env = {**os.environ, "USE_SYCL_KERNELS": "0"}
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKLOAD, xpu_clip, str(NUM_THREADS), str(ITERATIONS)],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"decoding hung for {TIMEOUT_SECONDS}s: {e.stderr}") from e

    assert result.returncode == 0, (
        f"decoding failed with exit code {result.returncode}:\n{result.stderr[-2000:]}"
    )

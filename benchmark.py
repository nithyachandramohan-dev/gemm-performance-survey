# benchmark.py

import torch
import torch.utils.benchmark as benchmark
import pandas as pd
import os
from config import MATRIX_SIZES, DTYPES, MIN_RUNTIME
from utils import get_device, get_dtype, calculate_tflops

def run_benchmark():

    device = get_device()
    results = []

    print(f"Running on device: {device}")

    for dtype_str in DTYPES:

        if device == "cpu" and dtype_str == "float16":
            continue

        dtype = get_dtype(dtype_str)

        for (M, N, K) in MATRIX_SIZES:

            print(f"Testing: {dtype_str} | Size: {M}x{N}x{K}")

            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(K, N, device=device, dtype=dtype)

            timer = benchmark.Timer(
                stmt="torch.mm(A, B)",
                globals={"A": A, "B": B}
            )

            measurement = timer.blocked_autorange(min_run_time=MIN_RUNTIME)
            time_taken = measurement.median

            tflops = calculate_tflops(M, N, K, time_taken)

            results.append({
                "device": device,
                "dtype": dtype_str,
                "M": M,
                "N": N,
                "K": K,
                "time_sec": time_taken,
                "TFLOPS": tflops
            })

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/gemm_results.csv", index=False)

    print("\nBenchmark completed.\n")
    print(df)

    return df
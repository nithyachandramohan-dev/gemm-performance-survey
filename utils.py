# utils.py

import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    return torch.float32

def calculate_tflops(M, N, K, time_sec):
    flops = 2 * M * N * K
    return flops / (time_sec * 1e12)
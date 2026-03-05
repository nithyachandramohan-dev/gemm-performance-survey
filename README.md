# PyTorch Micro-Benchmarking Suite for GEMM

## Overview
This project implements a PyTorch-based micro-benchmarking suite
to evaluate General Matrix Multiplication (GEMM) performance
across different matrix sizes, precisions, and hardware configurations.

## Features
- CPU and GPU benchmarking
- FP32 and FP16 precision comparison
- TFLOPS computation
- CSV result logging
- Performance visualization

## Formula

TFLOPS = (2 × M × N × K) / (time × 10^12)

## How to Run

Install dependencies:

pip install -r requirements.txt

Run benchmark:

python main.py

Results will be stored in:

results/gemm_results.csv
results/performance_plot.png

## Applications

Useful for:
- AI performance analysis
- Hardware evaluation
- Understanding compute-bound vs memory-bound workloads
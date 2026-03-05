# plot.py

import pandas as pd
import matplotlib.pyplot as plt

def generate_plot():

    df = pd.read_csv("results/gemm_results.csv")

    for dtype in df["dtype"].unique():
        subset = df[df["dtype"] == dtype]
        plt.plot(subset["M"], subset["TFLOPS"], marker='o', label=dtype)

    plt.xlabel("Matrix Size (M=N=K)")
    plt.ylabel("TFLOPS")
    plt.title("GEMM Performance Benchmark")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/performance_plot.png")
    plt.show()
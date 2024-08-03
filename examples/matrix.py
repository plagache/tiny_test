#!/usr/bin/env python3
import numpy as np
import time


def calculate_flops(a, b):
    giga = 1e9
    tera = 1e12
    # number of floating point operation is equal to S^2 with 2S for each
    flop = S * S * 2 * S
    # print(f"{flop/giga:.2f} GFLOP")

    st = time.monotonic()

    c = a @ b
    # print(c.shape, c.dtype, c)

    et = time.monotonic()

    delay = et - st

    # print(f"{flop / delay * (1 / tera):.2f} TFLOP/S")
    tflop = flop / delay * (1 / tera)
    return tflop


if __name__ == "__main__":
    S = 4096

    # S^2
    A = np.random.randn(S, S).astype(np.float32)
    # S^2
    B = np.random.randn(S, S).astype(np.float32)

    # tflop = calculate_flops(A, B)
    # print(f"{tflop:.2f} TFLOP/S")

    D = np.matrix([[1, 2], [3, 4]])
    M = np.matrix([[5, 6], [7, 8]])
    inner = np.inner(D, M)
    print(D)
    print(M)
    # print(inner)
    # print(D @ M)
    print(M @ D)

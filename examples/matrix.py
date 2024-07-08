#!/usr/bin/env python3
import numpy as np
import time

if __name__ == "__main__":

    S = 4096

    # S^2
    A = np.random.randn(S, S).astype(np.float32)
    # S^2
    B = np.random.randn(S, S).astype(np.float32)

    giga = 1e9
    tera = 1e12
    # number of floating point operation is equal to S^2 with 2S for each
    flop = S * S * 2 * S
    print(f"{flop/giga:.2f} GFLOP")

    st = time.monotonic()

    C = A @ B
    # print(C.shape, C.dtype, C)

    et = time.monotonic()

    delay = et - st

    print(f"{flop / delay * (1 / tera):.2f} TFLOP/S")

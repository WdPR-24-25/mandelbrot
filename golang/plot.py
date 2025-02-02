import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

files = sys.argv[1:]
assert len(files) > 0

fig, ax = plt.subplots()

for file in files:
    size, time = np.loadtxt(file).T
    ax.plot(size, time, "p-", label=Path("file").stem)

ax.set_title("Benchmark - Mandelbrot in Go")
ax.set_xlabel("Image Size [px]")
ax.set_ylabel(r"Time [$\mu$s]")

ax.set_xscale("log", base=2)
ax.set_yscale("log")

fig.legend()
fig.tight_layout()
fig.savefig(Path(__file__).parent / "benchmark.pdf")

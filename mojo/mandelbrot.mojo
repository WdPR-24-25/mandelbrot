from math import iota
from sys import num_physical_cores, simdwidthof, argv
from collections.string import atol, String

import benchmark
from algorithm import parallelize, vectorize
from complex import ComplexFloat64, ComplexSIMD
from memory import UnsafePointer


alias float_type = DType.float64
alias int_type = DType.int32
alias simd_width = 2 * simdwidthof[float_type]()
alias unit = benchmark.Unit.ms

alias size1 = 8
alias size2 = 16
alias size3 = 32
alias size4 = 64
alias size5 = 128
alias size6 = 256
alias size7 = 512
alias size8 = 1024
alias size9 = 2048
alias size10 = 4096
alias size11 = 8192

alias MAX_ITERS = 200

alias min_x = -2.1
alias max_x = 0.6
alias min_y = -1.2
alias max_y = 1.2


struct Matrix[type: DType, size: Int]:
    var data: UnsafePointer[Scalar[type]]

    fn __init__(out self):
        self.data = UnsafePointer[Scalar[type]].alloc(size * size)

    fn store[nelts: Int](self, row: Int, col: Int, val: SIMD[type, nelts]):
        self.data.store(row * size + col, val)

    fn save_to_ppm(self, filename: String) raises:
        """Save the Mandelbrot set as a PPM image file."""
        var file = open(filename, "w")
        # Write the PPM header
        file.write("P3\n")  # PPM magic number (text format)
        file.write(String(size) + " " + String(size) + "\n")  # Width and height
        file.write("255\n")  # Maximum color value

        # Write the pixel data
        for row in range(size):
            for col in range(size):
                var iterations = self.data.load(row * size + col)
                # Map iterations to a color (grayscale for simplicity)
                var color = iterations % 256 # Ensure color is in [0, 255]
                file.write(String(color) + " " + String(color) + " " + String(color) + " ")
            file.write("\n")
        file.close()


fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    var cx = c.re
    var cy = c.im
    var x = SIMD[float_type, simd_width](0)
    var y = SIMD[float_type, simd_width](0)
    var y2 = SIMD[float_type, simd_width](0)
    var iters = SIMD[int_type, simd_width](0)
    var t: SIMD[DType.bool, simd_width] = True

    for _ in range(MAX_ITERS):
        if not any(t):
            break
        y2 = y * y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)
    return iters


fn benchmark_for_size[size: Int]() raises:
    print("\nBenchmarking for size:", size)
    var matrix = Matrix[int_type, size]()

    @parameter
    fn worker(row: Int):
        alias scale_x = (max_x - min_x) / size
        alias scale_y = (max_y - min_y) / size

        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            var cy = min_y + row * SIMD[float_type, simd_width](scale_y)
            var c = ComplexSIMD[float_type, simd_width](cx, cy)
            matrix.store(row, col, mandelbrot_kernel_SIMD(c))

        vectorize[compute_vector, simd_width, size=size]()

    @parameter
    fn bench():
        for row in range(size):
            worker(row)

    @parameter
    fn bench_parallel():
        parallelize[worker](size, size)

    var vectorized = benchmark.run[bench](min_runtime_secs=10).mean(unit)
    print("Vectorized:", vectorized, unit)
    var parallelized = benchmark.run[bench_parallel](min_runtime_secs=10).mean(unit)
    print("Parallelized:", parallelized, unit)

    print("Parallel speedup:", vectorized / parallelized)

    # to generate image
    # var filename = "mandelbrot_size_" + String(size) + ".ppm"
    # matrix.save_to_ppm(filename)

    # to save times to file
    var filename = "mojo_benchmark_results.csv"
    var file = open(filename, "r")  # Open in append mode
    file_content = file.read()
    file.close()
    file = open(filename, "w")
    file.write(file_content)
    file.write(String(size) + "," + String(vectorized) + "," + String(parallelized) + "\n")
    file.close()
    matrix.data.free()


fn main() raises:
    # Initialize the CSV file with headers
    var filename = "mojo_benchmark_results.csv"
    var file = open(filename, "w")
    file.write("Size,Vectorized Time (ms),Parallelized Time (ms)\n")
    file.close()

    # MOJO SUPPORTS NO HYPERTHREADING I GUESS
    print("Number of physical cores:", num_physical_cores())

    benchmark_for_size[size1]()
    benchmark_for_size[size2]()
    benchmark_for_size[size3]()
    benchmark_for_size[size4]()
    benchmark_for_size[size5]()
    benchmark_for_size[size6]()
    benchmark_for_size[size7]()
    benchmark_for_size[size8]()
    benchmark_for_size[size9]()
    benchmark_for_size[size10]()
    benchmark_for_size[size11]()


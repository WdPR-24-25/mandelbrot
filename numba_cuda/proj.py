import os
import time
import csv
import numpy as np
from numba import njit, prange, cuda
from PIL import Image
import multiprocessing
import numba

numba.set_num_threads(multiprocessing.cpu_count())

SAVE_IMAGES = True

@njit
def mandelbrot_cpu_seq(width, height, x_min=-2.1, x_max=0.6, 
                         y_min=-1.2, y_max=1.2, max_iter=200):
    result = np.empty((height, width), dtype=np.int32)
    for j in range(height):
        y = y_min + j * (y_max - y_min) / (height - 1)
        for i in range(width):
            x = x_min + i * (x_max - x_min) / (width - 1)
            c_real = x
            c_imag = y
            z_real = 0.0
            z_imag = 0.0
            iter = 0
            while z_real*z_real + z_imag*z_imag <= 4.0 and iter < max_iter:
                temp = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = temp
                iter += 1
            result[j, i] = iter
    return result

@njit(parallel=True)
def mandelbrot_cpu_par(width, height, x_min=-2.1, x_max=0.6, 
                         y_min=-1.2, y_max=1.2, max_iter=200):
    result = np.empty((height, width), dtype=np.int32)
    for j in prange(height):
        y = y_min + j * (y_max - y_min) / (height - 1)
        for i in range(width):
            x = x_min + i * (x_max - x_min) / (width - 1)
            c_real = x
            c_imag = y
            z_real = 0.0
            z_imag = 0.0
            iter = 0
            while z_real*z_real + z_imag*z_imag <= 4.0 and iter < max_iter:
                temp = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = temp
                iter += 1
            result[j, i] = iter
    return result


@cuda.jit
def mandelbrot_gpu_par_kernel(result, width, height, 
                                x_min, x_max, y_min, y_max, max_iter):
    j, i = cuda.grid(2)
    if i < width and j < height:
        x = x_min + i * (x_max - x_min) / (width - 1)
        y = y_min + j * (y_max - y_min) / (height - 1)
        c_real = x
        c_imag = y
        z_real = 0.0
        z_imag = 0.0
        iter = 0
        while z_real*z_real + z_imag*z_imag <= 4.0 and iter < max_iter:
            temp = z_real*z_real - z_imag*z_imag + c_real
            z_imag = 2.0 * z_real * z_imag + c_imag
            z_real = temp
            iter += 1
        result[j, i] = iter

def mandelbrot_gpu_par(width, height, x_min=-2.1, x_max=0.6, 
                       y_min=-1.2, y_max=1.2, max_iter=200):
    result = np.empty((height, width), dtype=np.int32)
    d_result = cuda.to_device(result)
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    mandelbrot_gpu_par_kernel[blockspergrid, threadsperblock](d_result, width, height, x_min, x_max, y_min, y_max, max_iter)
    d_result.copy_to_host(result)
    return result

def save_image_pillow(data, filename, max_iter=200):
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    norm = np.uint8(255 * data / max_iter)
    img = Image.fromarray(norm, mode='L')
    img.save(filename)

def measure_time(func, size, repetitions, **kwargs):
    total_time = 0.0
    # "Rozgrzewka" – pierwsze wywołanie
    func(size, size, **kwargs)
    for _ in range(repetitions):
        start = time.perf_counter()
        func(size, size, **kwargs)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / repetitions


def main():
    max_iter = 200
    x_min, x_max = -2.1, 0.6
    y_min, y_max = -1.2, 1.2

    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    repetitions = 10  

    if SAVE_IMAGES:
        for folder in ["images_cpu_seq", "images_cpu_par", "images_gpu_par"]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    csv_filename = "comparison_times.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "CPU_seq", "CPU_par", "GPU_par"])
        for size in sizes:
            cpu_seq_time = measure_time(mandelbrot_cpu_seq, size, repetitions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, max_iter=max_iter)
            cpu_par_time = measure_time(mandelbrot_cpu_par, size, repetitions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, max_iter=max_iter)
            gpu_par_time = measure_time(mandelbrot_gpu_par, size, repetitions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, max_iter=max_iter)
            
            print(f"Size {size}: CPU_seq {cpu_seq_time:.6f}s, CPU_par {cpu_par_time:.6f}s, GPU_par {gpu_par_time:.6f}s")
            
            writer.writerow([size, cpu_seq_time, cpu_par_time, gpu_par_time])
            
            if SAVE_IMAGES:
                img_cpu_seq = mandelbrot_cpu_seq(size, size, x_min, x_max, y_min, y_max, max_iter)
                save_image_pillow(img_cpu_seq, os.path.join("images_cpu_seq", f"mandelbrot_cpu_seq_{size}.png"), max_iter)
                
                img_cpu_par = mandelbrot_cpu_par(size, size, x_min, x_max, y_min, y_max, max_iter)
                save_image_pillow(img_cpu_par, os.path.join("images_cpu_par", f"mandelbrot_cpu_par_{size}.png"), max_iter)
                  
                img_gpu_par = mandelbrot_gpu_par(size, size, x_min, x_max, y_min, y_max, max_iter)
                save_image_pillow(img_gpu_par, os.path.join("images_gpu_par", f"mandelbrot_gpu_par_{size}.png"), max_iter)
    
    print(f"Wyniki zapisane do {csv_filename}")

if __name__ == "__main__":
    main()

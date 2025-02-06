#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Dołączamy bibliotekę do zapisywania obrazów

#define MAX_ITERATIONS 200


// Kernel CUDA do obliczania zbioru Mandelbrota
__global__ void mandelbrotKernel(int *image, int width, int height, double xMin, double yMin, double xMax, double yMax, int maxIterations) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < width && py < height) {
        double x0 = xMin + px * (xMax - xMin) / width;
        double y0 = yMin + py * (yMax - yMin) / height;

        double x = 0.0, y = 0.0;
        int iteration = 0;
        while (x * x + y * y <= 4 && iteration < maxIterations) {
            double xTemp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = xTemp;
            iteration++;
        }
        int i = 255*iteration/maxIterations;
        int color = (i << 16) | (i << 8) | i;
        image[py * width + px] = color;
    }
}

// Funkcja do generowania i zapisywania obrazu
void generateAndSaveImage(const char *filePath, int width, int height, double xMin, double yMin, double xMax, double yMax, int maxIterations) {
    int *image = (int *)malloc(width * height * sizeof(int));
    int *d_image;
    cudaMalloc((void **)&d_image, width * height * sizeof(int));

    dim3 blockSize(32, 32); // Optymalny rozmiar bloku
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, width, height, xMin, yMin, xMax, yMax, maxIterations);
    cudaDeviceSynchronize(); // Poczekaj na zakończenie kernela

    cudaMemcpy(image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Konwersja obrazu do formatu RGB (3 kanały)
    unsigned char *rgbImage = (unsigned char *)malloc(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int color = image[y * width + x];
            rgbImage[(y * width + x) * 3 + 0] = (color >> 16) & 0xFF; // R
            rgbImage[(y * width + x) * 3 + 1] = (color >> 8) & 0xFF;  // G
            rgbImage[(y * width + x) * 3 + 2] = color & 0xFF;         // B
        }
    }

    // Zapisz obraz do pliku PNG
    stbi_write_png(filePath, width, height, 3, rgbImage, width * 3);

    free(image);
    free(rgbImage);
    cudaFree(d_image);

    printf("Obraz zapisany: %s\n", filePath);
}

int main() {
    int width = 8192;  // Szerokość obrazu
    int height = 8192; // Wysokość obrazu
    double xMin = -2.1;
    double yMin = -1.2;
    double xMax = 0.6;
    double yMax = 1.2;
    int maxIterations = MAX_ITERATIONS;

    // Generuj i zapisz obraz
    generateAndSaveImage("mandelbrot.png", width, height, xMin, yMin, xMax, yMax, maxIterations);

    return 0;
}
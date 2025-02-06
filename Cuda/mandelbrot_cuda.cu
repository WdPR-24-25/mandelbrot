#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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

        int i = iteration/MAX_ITERATIONS;
        int color = (i << 16) | (i << 8) | i;
        image[py * width + px] = color;
    }
}

// Funkcja do pomiaru średniego czasu wykonania kernela
double measureAverageTime(int width, int height, double xMin, double yMin, double xMax, double yMax, int maxIterations, int repetitions) {
    int *image = (int *)malloc(width * height * sizeof(int));
    int *d_image;
    cudaMalloc((void **)&d_image, width * height * sizeof(int));

    dim3 blockSize(32, 32); // Optymalny rozmiar bloku
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0.0f;

    for (int i = 0; i < repetitions; i++) {
        cudaEventRecord(start); // Rozpocznij pomiar czasu

        mandelbrotKernel<<<gridSize, blockSize>>>(d_image, width, height, xMin, yMin, xMax, yMax, maxIterations);
        cudaDeviceSynchronize(); // Poczekaj na zakończenie kernela

        cudaEventRecord(stop); // Zakończ pomiar czasu
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); // Oblicz czas w milisekundach
        totalTime += elapsedTime;
    }

    cudaMemcpy(image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    free(image);
    cudaFree(d_image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalTime / repetitions; // Średni czas w milisekundach
}

// Funkcja do zapisywania wyników do pliku CSV
void measureAndSaveAllTimes(int *sizes, int numSizes, double xMin, double yMin, double xMax, double yMax, int maxIterations, int repetitions, const char *csvFilePath) {
    FILE *csvFile = fopen(csvFilePath, "w");
    if (!csvFile) {
        fprintf(stderr, "Error opening CSV file: %s\n", csvFilePath);
        return;
    }

    fprintf(csvFile, "Size (px),Average Time (ms)\n");

    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];
        double averageTime = measureAverageTime(size, size, xMin, yMin, xMax, yMax, maxIterations, repetitions);
        fprintf(csvFile, "%d,%.6f\n", size, averageTime);
        printf("Size: %d, Average Time: %.6f ms\n", size, averageTime);
    }

    fclose(csvFile);
    printf("Results saved to: %s\n", csvFilePath);
}

int main() {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    double xMin = -2.1;
    double yMin = -1.2;
    double xMax = 0.6;
    double yMax = 1.2;
    int maxIterations = MAX_ITERATIONS;
    int repetitions = 10;

    // Pomiar i zapis czasów do pliku CSV
    measureAndSaveAllTimes(sizes, numSizes, xMin, yMin, xMax, yMax, maxIterations, repetitions, "mandelbrot_cuda_times.csv");

    return 0;
}
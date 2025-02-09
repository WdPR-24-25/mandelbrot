#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <cstdlib>

using namespace std;
using namespace std::chrono;


const int maxIterations = 200;
const double xMin = -2.1, xMax = 0.6, yMin = -1.2, yMax = 1.2;


void generateMandelbrotPart(int width, int height, int startY, int endY, vector<vector<int>>& image) {
    for (int py = startY; py < endY; py++) {
        for (int px = 0; px < width; px++) {
            double x0 = xMin + (xMax - xMin) * px / width;
            double y0 = yMin + (yMax - yMin) * py / height;

            int iteration = 0;
            double x = 0, y = 0;

            while (x * x + y * y <= 4 && iteration < maxIterations) {
                double xTemp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = xTemp;
                iteration++;
            }

            image[py][px] = iteration;
        }
    }
}


void generateMandelbrotParallel(int width, int height, const string &filename) {
    vector<vector<int>> image(height, vector<int>(width));

    int numThreads = thread::hardware_concurrency(); // Liczba rdzeni CPU
    vector<thread> threads;

    int rowsPerThread = height / numThreads;

    for (int t = 0; t < numThreads; t++) {
        int startY = t * rowsPerThread;
        int endY = (t == numThreads - 1) ? height : (t + 1) * rowsPerThread;
        threads.push_back(thread(generateMandelbrotPart, width, height, startY, endY, ref(image)));
    }

    for (auto& t : threads) {
        t.join();
    }

    
    ofstream file(filename);
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            file << image[py][px] << " ";
        }
        file << endl;
    }
    file.close();
}


void runPerformanceTestsParallel(const vector<int>& sizes, int repetitions) {
    for (int size : sizes) {
        long long totalTime = 0;

        for (int i = 0; i < repetitions; i++) {
            auto start = high_resolution_clock::now();
            generateMandelbrotParallel(size, size, "mandelbrot_" + to_string(size) + "x" + to_string(size) + "_" + to_string(i) + ".txt");
            auto end = high_resolution_clock::now();
            totalTime += duration_cast<nanoseconds>(end - start).count();
        }

        long long averageTime = totalTime / repetitions;
        cout << "Average time for " << size << "x" << size << ": " << averageTime << " nanoseconds" << endl;
    }
}

int main() {
    vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int repetitions = 5;

    runPerformanceTestsParallel(sizes, repetitions);

    return 0;
}

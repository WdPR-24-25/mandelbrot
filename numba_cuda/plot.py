import matplotlib.pyplot as plt
import csv
import os

def read_csv(filename, encoding="latin-1"):
    sizes = []
    cpu_seq_times = []
    cpu_par_times = []
    gpu_par_times = []
    with open(filename, 'r', newline='', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        print("Nagłówki w CSV:", reader.fieldnames)
        size_key = "Size" if "Size" in reader.fieldnames else "Rozmiar"
        for row in reader:
            try:
                sizes.append(int(row[size_key]))
                cpu_seq_times.append(float(row["CPU_seq"]))
                cpu_par_times.append(float(row["CPU_par"]))
                gpu_par_times.append(float(row["GPU_par"]))
            except ValueError:
                continue
    return sizes, cpu_seq_times, cpu_par_times, gpu_par_times

csv_filename = "comparison_times.csv"
if not os.path.exists(csv_filename):
    print(f"Plik CSV '{csv_filename}' nie został znaleziony. Upewnij się, że dane są zapisane.")
    exit(1)

sizes, cpu_seq, cpu_par, gpu_par = read_csv(csv_filename, encoding="latin-1")
cpu_seq = [t * 1e6 for t in cpu_seq]
cpu_par = [t * 1e6 for t in cpu_par]
gpu_par = [t * 1e6 for t in gpu_par]

plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_seq, marker='o', linestyle='-', color='blue', label='CPU Seq')
plt.plot(sizes, cpu_par, marker='o', linestyle='-', color='green', label='CPU Par')
plt.plot(sizes, gpu_par, marker='o', linestyle='-', color='red', label='GPU Par')
plt.xlabel("Rozmiar (piksele)")
plt.ylabel("Średni czas (μs)")
plt.title("Mandelbrot czasy")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.tight_layout()
output_plot = "comparison_plot.png"
plt.savefig(output_plot, dpi=300)
print(f"Wykres zapisany do: {output_plot}")
plt.show()

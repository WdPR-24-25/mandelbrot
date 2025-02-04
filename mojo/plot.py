import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "mojo_benchmark_results.csv"
data = pd.read_csv(csv_file)

# Extract columns
sizes = data["Size"]
vectorized_times = data["Vectorized Time (ms)"]
parallelized_times = data["Parallelized Time (ms)"]

# Create a plot with logarithmic scale
plt.figure(figsize=(10, 6))

# Plot vectorized and parallelized times
plt.plot(sizes, vectorized_times, marker='o', label="Vectorized Time (ms)", linestyle='-', color='blue')
plt.plot(sizes, parallelized_times, marker='s', label="Parallelized Time (ms)", linestyle='--', color='red')

# Set logarithmic scale for the x-axis
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.xlabel("Matrix Size (log scale)")
plt.ylabel("Time (ms, log scale)")
plt.title("Benchmark Times: Vectorized vs Parallelized")
plt.legend()

# Add grid for better readability
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Save the plot as PNG and PDF
output_png = "benchmark_times_log_scale.png"
output_pdf = "benchmark_times_log_scale.pdf"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

# Show the plot
plt.show()
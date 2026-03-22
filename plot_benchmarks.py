import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# 1. Load and Prepare the Data
# =============================================================================
CSV_FILE = 'benchmarks_comprehensive.csv'

if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Run benchmark.sh first.")
    exit(1)

# Read data and filter out any failed runs
df = pd.read_csv(CSV_FILE)
df = df[df['runtime_sec'] != 'ERROR'].copy()
df['runtime_sec'] = df['runtime_sec'].astype(float)

# Aggregate data: mean and standard deviation across the 5 seeds
agg_df = df.groupby(['executable', 'size', 'generations', 'threads'])['runtime_sec'].agg(['mean', 'std']).reset_index()

# Map executable names to readable labels for plots
labels = {
    'gs': 'Grid Sequential',
    'as': 'Active Sequential',
    'go': 'Grid OpenMP (Optimized)',
    'ao': 'Active OpenMP (Optimized)',
}

# Define color schemes
colors = {'go': '#1f77b4',
          'ao': '#ff7f0e',
          'gs': '#2ca02c',
          'as': '#d62728'}

# =============================================================================
# Plot 1: Real vs. Ideal Speedup (Strong Scaling)
# =============================================================================
def plot_speedup():
    plt.figure(figsize=(10, 6))

    # We will look at the largest workload for the best scaling curves
    target_size = agg_df['size'].max()
    target_gens = agg_df['generations'].max()

    subset = agg_df[(agg_df['size'] == target_size) & (agg_df['generations'] == target_gens)]

    parallel_exes = ['go', 'ao']
    max_threads = subset['threads'].max()

    for exe in parallel_exes:
        data = subset[subset['executable'] == exe].sort_values('threads')
        if data.empty:
            continue

        # T1 is the runtime at 1 thread
        t1 = data[data['threads'] == 1]['mean'].values[0]
        speedup = t1 / data['mean']

        plt.plot(data['threads'], speedup, marker='o', linewidth=2,
                 label=f"{labels[exe]}", color=colors[exe])

    # Plot Ideal Speedup (y = x)
    ideal_x = np.arange(1, max_threads + 1)
    plt.plot(ideal_x, ideal_x, linestyle='--', color='black', alpha=0.6, label='Ideal Speedup (Linear)')

    plt.title(f'OpenMP Speedup Analysis (N={target_size}, Gens={target_gens})', fontsize=14, pad=15)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup Factor ($T_1 / T_n$)', fontsize=12)
    plt.xticks(sorted(subset['threads'].unique()))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plot_1_speedup.png', dpi=300)
    print("Saved plot_1_speedup.png")

# =============================================================================
# Plot 2: Algorithmic Comparison (Dense vs. Sparse over Grid Size)
# =============================================================================
def plot_algorithmic_crossover():
    plt.figure(figsize=(10, 6))

    target_gens = agg_df['generations'].max()
    # Use 1 thread to purely compare the algorithms without multithreading noise
    subset = agg_df[(agg_df['generations'] == target_gens) & (agg_df['threads'] == 1)]

    compare_exes = ['gs', 'as']

    for exe in compare_exes:
        data = subset[subset['executable'] == exe].sort_values('size')
        if data.empty:
            continue

        plt.plot(data['size'], data['mean'], marker='s', linewidth=2,
                 label=labels[exe], color=colors[exe])

    plt.title(f'Algorithmic Complexity: Dense vs Sparse (Gens={target_gens}, 1 Thread)', fontsize=14, pad=15)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Runtime (Seconds)', fontsize=12)
    plt.yscale('log') # Log scale because N^3 grows massively
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plot_2_algo_crossover.png', dpi=300)
    print("Saved plot_2_algo_crossover.png")

# =============================================================================
# Plot 3: OpenMP Runtime Comparison
# =============================================================================
def plot_optimization_impact():
    plt.figure(figsize=(10, 6))

    target_size = agg_df['size'].max()
    target_gens = agg_df['generations'].max()
    subset = agg_df[(agg_df['size'] == target_size) & (agg_df['generations'] == target_gens)]

    compare_exes = ['go', 'ao']

    for exe in compare_exes:
        data = subset[subset['executable'] == exe].sort_values('threads')
        if data.empty:
            continue

        plt.plot(data['threads'], data['mean'], marker='o', linestyle='-',
                 linewidth=2, label=labels[exe], color=colors[exe])

    plt.title(f'OpenMP Runtime Comparison (N={target_size}, Gens={target_gens})', fontsize=14, pad=15)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Runtime (Seconds)', fontsize=12)
    plt.xticks(sorted(subset['threads'].unique()))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plot_3_optimizations.png', dpi=300)
    print("Saved plot_3_optimizations.png")

# =============================================================================
# Execute
# =============================================================================
plot_speedup()
plot_algorithmic_crossover()
plot_optimization_impact()

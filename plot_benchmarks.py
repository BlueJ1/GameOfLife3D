import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data
try:
    df = pd.read_csv('benchmarks.csv')
except FileNotFoundError:
    print("Error: benchmarks.csv not found. Run your bash script first.")
    exit()

# 2. Calculate Speedup relative to 1 thread (T1 / Tn)
t1 = df.loc[df['threads'] == 1, 'runtime_sec'].values[0]
df['speedup'] = t1 / df['runtime_sec']

# 3. Create the visualization
plt.figure(figsize=(10, 7))

# --- Plot Actual Measured Speedup ---
plt.plot(df['threads'], df['speedup'], 
         marker='o', linestyle='-', color='#1f77b4', 
         linewidth=2, label='Measured Speedup')

# --- Plot Ideal Speedup (The reference line) ---
# This is a y=x line representing 100% parallel efficiency
plt.plot(df['threads'], df['threads'], 
         linestyle='--', color='red', alpha=0.8, 
         linewidth=1.5, label='Ideal Speedup (Linear)')

# 4. Final Plot Styling
plt.title('OpenMP Scaling Performance: 3-D Game of Life', fontsize=14, pad=15)
plt.xlabel('Number of Threads (Cores)', fontsize=12)
plt.ylabel('Speedup Factor ($T_1 / T_n$)', fontsize=12)

# Ensure the axes are equal for clear visual comparison
plt.xlim(0, max(df['threads']) + 1)
plt.ylim(0, max(df['threads']) + 1)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# 5. Save the output
output_filename = 'speedup_analysis.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300)

print(f"Plot successfully saved to: {output_filename}")

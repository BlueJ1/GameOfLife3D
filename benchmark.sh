#!/bin/bash

# Configuration
EXE="./ao"
SIZE=200
GENS=20
SEED=42
OUTPUT_CSV="benchmarks.csv"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "Error: $EXE not found. Please compile your code first (e.g., gcc -O3 -fopenmp grid_openmp.c -o go)"
    exit 1
fi

# Prepare CSV header
echo "threads,runtime_sec" > "$OUTPUT_CSV"

echo "Starting benchmarks for cores 1 to 40..."
echo "---------------------------------------"

for cores in {1..16}
do
    # Run the command and capture the output
    # We use 'tail -n 2' and 'head -n 1' to grab the specific line containing the time
    result=$($EXE $SIZE $GENS $SEED $cores)

    # Extract the numerical time value using grep/sed
    runtime=$(echo "$result" | grep "Simulation time" | awk '{print $3}')

    # Save to CSV and print to terminal
    echo "$cores,$runtime" >> "$OUTPUT_CSV"
    echo "Cores: $cores | Runtime: ${runtime}s"
done

echo "---------------------------------------"
echo "Benchmarking complete. Results saved to $OUTPUT_CSV"

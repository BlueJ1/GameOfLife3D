#!/bin/bash

# ==============================================================================
# Configuration Section
# ==============================================================================

# List of executables to benchmark
# gs = grid_sequential, as = active_sequential
# go = grid_openmp, go2 = grid_openmp_v0
# ao = active_omp, ao2 = active_omp_v0
EXECUTABLES=("gs" "as" "go" "go2" "ao" "ao2")

# Grid sizes (N x N x N)
SIZES=(50 100 200)

# Number of generations to simulate
GENERATIONS=(20 50)

# Thread counts to test for OpenMP parallel versions
THREADS=(1 2 3 4 6 8 12 16 24 32 48 64)

# 5 distinct seeds for statistical averaging
SEEDS=(42 123 999)

# Output file
OUTPUT_CSV="benchmarks_comprehensive.csv"

# ==============================================================================

# 1. Verification and Setup
echo "Checking for executables..."
for exe in "${EXECUTABLES[@]}"; do
    if [ ! -f "./$exe" ]; then
        echo "Error: ./$exe not found. Running 'make all'..."
        make all
        if [ $? -ne 0 ]; then
            echo "Make failed. Please fix compilation errors."
            exit 1
        fi
        break
    fi
done

# Initialize CSV header
echo "executable,size,generations,threads,seed,runtime_sec" > "$OUTPUT_CSV"

echo "======================================================="
echo " Starting Comprehensive Benchmarks"
echo " Executables: ${EXECUTABLES[*]}"
echo " Sizes:       ${SIZES[*]}"
echo " Generations: ${GENERATIONS[*]}"
echo " Threads:     ${THREADS[*]}"
echo " Seeds:       ${SEEDS[*]}"
echo " Output:      $OUTPUT_CSV"
echo "======================================================="

# 2. Nested Benchmarking Loops
for exe in "${EXECUTABLES[@]}"; do
    echo ">> Benchmarking $exe..."

    for size in "${SIZES[@]}"; do
        for gens in "${GENERATIONS[@]}"; do
            for threads in "${THREADS[@]}"; do

                # OPTIMIZATION: Sequential code doesn't use threads.
                # Run it only once for 'threads=1' and skip the rest to save time.
                if [[ "$exe" == "gs" || "$exe" == "as" ]]; then
                    if [ "$threads" -ne 1 ]; then
                        continue
                    fi
                fi

                for seed in "${SEEDS[@]}"; do

                    # Execute and capture output
                    # The sequential versions will simply ignore the 4th argument (threads)
                    result=$(./$exe $size $gens $seed $threads 2>&1)

                    # Extract the numerical time value
                    runtime=$(echo "$result" | grep "Simulation time" | awk '{print $3}')

                    # Handle potential errors (if runtime is empty)
                    if [ -z "$runtime" ]; then
                        echo "   [ERROR] Run failed: ./$exe $size $gens $seed $threads"
                        runtime="ERROR"
                    else
                        echo "   [OK] $exe | Size: $size | Gens: $gens | Threads: $threads | Seed: $seed | Time: ${runtime}s"
                    fi

                    # Write to CSV
                    echo "$exe,$size,$gens,$threads,$seed,$runtime" >> "$OUTPUT_CSV"

                done
            done
        done
    done
done

echo "======================================================="
echo "Benchmarking complete! Results saved to $OUTPUT_CSV"

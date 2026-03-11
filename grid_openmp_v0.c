#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Include OpenMP header

// Function to get the 1D index from 3D coordinates
int get_index(int x, int y, int z, int size) {
    return (x * size * size) + (y * size) + z;
}

// Function to count living neighbors using a wrapping (toroidal) boundary
int count_neighbors(int *grid, int size, int x, int y, int z) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                // Use modulo to wrap around the edges safely
                int nx = (x + dx + size) % size;
                int ny = (y + dy + size) % size;
                int nz = (z + dz + size) % size;

                count += grid[get_index(nx, ny, nz, size)];
            }
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int size = 30;        // Default grid size
    int generations = 20; // Default number of generations
    int seed = 42;        // Default deterministic seed

    // Check for grid size argument
    if (argc > 1) {
        size = atoi(argv[1]);
        if (size < 5) {
            printf("Size too small. Please use a size of at least 5.\n");
            return 1;
        }
    }

    // Check for generation count argument
    if (argc > 2) {
        generations = atoi(argv[2]);
        if (generations < 1) {
            printf("Generations must be at least 1.\n");
            return 1;
        }
    }

    // Check for seed argument
    if (argc > 3) {
        seed = atoi(argv[3]);
    }

    printf("Initializing a %dx%dx%d universe for %d generations (Seed: %d)...\n", size, size, size, generations, seed);

    // Dynamically allocate memory for the grids using calloc (initializes to 0)
    int total_cells = size * size * size;
    int *grid = (int *)calloc(total_cells, sizeof(int));
    int *next_grid = (int *)calloc(total_cells, sizeof(int));

    if (grid == NULL || next_grid == NULL) {
        printf("Memory allocation failed! Grid size might be too large.\n");
        return 1;
    }

    FILE *file = fopen("evolution.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        free(grid);
        free(next_grid);
        return 1;
    }

    // Initialize a random "primordial soup" deterministically
    srand((unsigned int)seed);
    int center = size / 2;
    for (int i = center - 3; i <= center + 3; i++) {
        for (int j = center - 3; j <= center + 3; j++) {
            for (int k = center - 3; k <= center + 3; k++) {
                grid[get_index(i, j, k, size)] = rand() % 2;
            }
        }
    }

    // Run the simulation
    for (int gen = 0; gen < generations; gen++) {
        fprintf(file, "=== Generation %d ===\n", gen);

        // 1. Sequential Logging Phase
        // Keep file I/O sequential to prevent race conditions and jumbled text
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                for (int z = 0; z < size; z++) {
                    int current_idx = get_index(x, y, z, size);
                    if (grid[current_idx] == 1) {
                        fprintf(file, "(%d, %d, %d)\n", x, y, z);
                    }
                }
            }
        }

        // 2. Parallel Computation Phase
        // Use collapse(3) to merge the nested loops into one large parallel loop
        #pragma omp parallel for collapse(3)
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                for (int z = 0; z < size; z++) {
                    
                    int current_idx = get_index(x, y, z, size);
                    
                    // Apply Rule 4555
                    int neighbors = count_neighbors(grid, size, x, y, z);

                    if (grid[current_idx] == 1) {
                        if (neighbors == 4 || neighbors == 5) {
                            next_grid[current_idx] = 1;
                        }
                    } else {
                        if (neighbors == 5) {
                            next_grid[current_idx] = 1;
                        }
                    }
                }
            }
        }

        // 3. Parallel Memory Update Phase
        // Copy next_grid to grid and clear next_grid for the next loop
        #pragma omp parallel for
        for (int i = 0; i < total_cells; i++) {
            grid[i] = next_grid[i];
            next_grid[i] = 0; // Reset next_grid for the next generation
        }

        fprintf(file, "\n");
    }

    fclose(file);

    // Free the dynamically allocated memory
    free(grid);
    free(next_grid);

    printf("Simulation complete. Results saved to evolution.txt\n");

    return 0;
}

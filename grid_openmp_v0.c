#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Include OpenMP header
#include <stdarg.h>

/* ------------------------------------------------------------------ */
/*  Dynamic string buffer for deferred file output                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb, size_t cap) {
    if (cap < 4096) cap = 4096;
    sb->data = malloc(cap);
    sb->len  = 0;
    sb->cap  = cap;
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) return;
    while (sb->len + (size_t)needed + 1 > sb->cap) {
        sb->cap *= 2;
        sb->data = realloc(sb->data, sb->cap);
    }
    va_start(ap, fmt);
    vsnprintf(sb->data + sb->len, (size_t)needed + 1, fmt, ap);
    va_end(ap);
    sb->len += (size_t)needed;
}

static void sb_free(StrBuf *sb) { free(sb->data); }

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

    /* --- In-memory output buffer (written to file at the end) --- */
    StrBuf out_buf;
    sb_init(&out_buf, (size_t)total_cells * 15 * (size_t)generations);

    // Initialize a random "primordial soup" over the entire N³ universe
    srand((unsigned int)seed);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                grid[get_index(i, j, k, size)] = rand() % 2;
            }
        }
    }

    // Run the simulation
    for (int gen = 0; gen < generations; gen++) {
        sb_printf(&out_buf, "=== Generation %d ===\n", gen);

        // 1. Sequential Logging Phase
        // Buffer output to write at the end
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                for (int z = 0; z < size; z++) {
                    int current_idx = get_index(x, y, z, size);
                    if (grid[current_idx] == 1) {
                        sb_printf(&out_buf, "(%d, %d, %d)\n", x, y, z);
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

        sb_printf(&out_buf, "\n");
    }

    // Write all output to file at the very end
    FILE *file = fopen("evolution.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        sb_free(&out_buf);
        free(grid);
        free(next_grid);
        return 1;
    }
    fwrite(out_buf.data, 1, out_buf.len, file);
    fclose(file);

    sb_free(&out_buf);

    // Free the dynamically allocated memory
    free(grid);
    free(next_grid);

    printf("Simulation complete. Results saved to evolution.txt\n");

    return 0;
}

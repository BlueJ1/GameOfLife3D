/*
 * 3-D Game of Life – OpenMP-parallelised version (optimised)
 *
 * Key optimisations over the naïve parallel version:
 *   1. unsigned char grid (1 B/cell) instead of int (4 B/cell)  → 4× less
 *      memory, much better cache utilisation for large grids.
 *   2. Interior / boundary split – the expensive modulo wrapping is only
 *      used for the thin shell of boundary cells; interior cells use plain
 *      addition.
 *   3. Pointer swap instead of a full-array copy each generation.
 *   4. static inline helpers so the compiler can inline and optimise the
 *      hot neighbour-counting path.
 *   5. Larger stdio buffer (1 MiB) to reduce write-syscall overhead.
 *   6. Wall-clock timing with omp_get_wtime() for easy benchmarking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* 3-D → 1-D index (row-major: x varies slowest, z fastest) */
static inline int idx(int x, int y, int z, int ssq, int s) {
    return x * ssq + y * s + z;
}

/* Count 26 neighbours – INTERIOR cell (no coordinate wraps) */
static inline int count_interior(const unsigned char *restrict g,
                                 int x, int y, int z,
                                 int ssq, int s) {
    int c = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int bx = (x + dx) * ssq;
        for (int dy = -1; dy <= 1; dy++) {
            int bxy = bx + (y + dy) * s;
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                c += g[bxy + (z + dz)];
            }
        }
    }
    return c;
}

/* Count 26 neighbours – BOUNDARY cell (toroidal / wrapping) */
static inline int count_boundary(const unsigned char *restrict g,
                                 int x, int y, int z,
                                 int ssq, int s) {
    int c = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int bx = ((x + dx + s) % s) * ssq;
        for (int dy = -1; dy <= 1; dy++) {
            int bxy = bx + ((y + dy + s) % s) * s;
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                c += g[bxy + ((z + dz + s) % s)];
            }
        }
    }
    return c;
}

/* Rule 4555: survive on 4 or 5 neighbours, birth on exactly 5 */
static inline unsigned char apply_rule(unsigned char alive, int nbrs) {
    if (alive)
        return (nbrs == 4 || nbrs == 5) ? 1 : 0;
    else
        return (nbrs == 5) ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    int size        = 30;   /* default grid size        */
    int generations = 20;   /* default generation count  */
    int seed        = 42;   /* default RNG seed          */
    int num_threads = 0;    /* 0 = let OpenMP decide     */

    /* Parse CLI arguments: size generations seed threads */
    if (argc > 1) {
        size = atoi(argv[1]);
        if (size < 5) { puts("Size too small. Please use a size of at least 5."); return 1; }
    }
    if (argc > 2) {
        generations = atoi(argv[2]);
        if (generations < 1) { puts("Generations must be at least 1."); return 1; }
    }
    if (argc > 3) {
        seed = atoi(argv[3]);
    }
    if (argc > 4) {
        num_threads = atoi(argv[4]);
        if (num_threads < 1) { puts("Thread count must be at least 1."); return 1; }
        omp_set_num_threads(num_threads);
    }

    /* Report actual thread count (query inside a parallel region) */
    int actual_threads;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    printf("Initializing a %dx%dx%d universe for %d generations (Seed: %d, Threads: %d)...\n",
           size, size, size, generations, seed, actual_threads);

    const int total_size = size * size * size;
    const int size_squared   = size * size;              /* precomputed size^2 */

    /*
     * Allocate grids as unsigned char (1 byte per cell) instead of int
     * (4 bytes).  For a 100³ grid this is 1 MB vs. 4 MB — a big win for
     * cache-line utilisation and memory-bandwidth pressure.
     */
    unsigned char *grid      = calloc(total_size, 1);
    unsigned char *next_grid = calloc(total_size, 1);
    if (!grid || !next_grid) { puts("Memory allocation failed!"); return 1; }

    /* Open output file with a large buffer to cut down on write syscalls */
    FILE *file = fopen("evolution.txt", "w");
    if (!file) { puts("Error opening file!"); free(grid); free(next_grid); return 1; }
    setvbuf(file, NULL, _IOFBF, 1 << 20);      /* 1 MiB write buffer */

    /* Seed the "primordial soup" over the entire N³ universe */
    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                grid[idx(i, j, k, size_squared, size)] = (unsigned char)(rand() % 2);

    double t0 = omp_get_wtime();

    /* ============================================================== */
    /*  Simulation loop                                                */
    /* ============================================================== */
    for (int gen = 0; gen < generations; gen++) {

        /* --- 1. Sequential logging (file order must be deterministic) --- */
        fprintf(file, "=== Generation %d ===\n", gen);
        for (int i = 0; i < total_size; i++) {
            if (grid[i]) {
                int x = i / size_squared;
                int r = i % size_squared;
                fprintf(file, "(%d, %d, %d)\n", x, r / size, r % size);
            }
        }

        /* --- 2. Parallel computation -------------------------------- */
        /*
         * One flat loop over every cell.  Interior cells (all coords
         * strictly inside [1 .. size-2]) use the fast, modulo-free
         * neighbour count; boundary cells fall back to the wrapping
         * version.  Branch prediction handles the split efficiently
         * since the vast majority of cells are interior.
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_size; i++) {
            int x = i / size_squared;
            int r = i % size_squared;
            int y = r / size;
            int z = r % size;

            int nbrs;
            if (x > 0 && x < size - 1 &&
                y > 0 && y < size - 1 &&
                z > 0 && z < size - 1)
                nbrs = count_interior(grid, x, y, z, size_squared, size);
            else
                nbrs = count_boundary(grid, x, y, z, size_squared, size);

            next_grid[i] = apply_rule(grid[i], nbrs);
        }

        /* --- 3. Pointer swap (O(1)) + clear next buffer ------------- */
        unsigned char *tmp = grid;
        grid      = next_grid;
        next_grid = tmp;
        memset(next_grid, 0, (size_t)total_size);    /* fast library memset */

        fprintf(file, "\n");
    }

    double t1 = omp_get_wtime();
    printf("Simulation time: %.4f s\n", t1 - t0);

    fclose(file);
    free(grid);
    free(next_grid);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

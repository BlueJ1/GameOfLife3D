/*
 * 3-D Game of Life 4555 — Active-cell flat-array OpenMP version
 *
 * Uses flat arrays for neighbour counts (per-thread private, no atomics)
 * and alive status, eliminating all hash-table overhead.  A fused
 * reduce + rule-evaluation pass produces the next generation.
 *
 * The thread team is created once and reused across all generations,
 * minimising fork/join overhead.
 *
 * Per generation:
 *   1. Sort + log living cells                              (single)
 *   2. Scatter: mark alive, increment neighbour counts      (parallel)
 *   3. Reduce + evaluate + clear: sum counts, apply rule,   (parallel)
 *      collect survivors/births, zero arrays for next gen
 *   4. Concatenate per-thread results, swap living lists     (single)
 *
 * Complexity per generation: O(26p/T + N³/T)  where p = live-cell count,
 * T = thread count.  For small-to-medium N (≤ ~200) this outperforms
 * hash-based approaches due to zero hashing overhead and perfect
 * cache-line utilisation.
 *
 * Rule 4555:  alive → alive iff nbrs ∈ {4,5};  dead → alive iff nbrs == 5.
 * Boundary: toroidal.
 *
 * Compile:
 *   gcc -O3 -fopenmp -o active_omp active_omp.c
 *
 * Usage:
 *   ./active_omp [size] [generations] [seed] [threads]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ================================================================== */
/*  Coordinate type + dynamic list                                     */
/* ================================================================== */

typedef struct { int x, y, z; } Coord;

typedef struct {
    Coord *data;
    int    count;
    int    capacity;
} CoordList;

static void cl_init(CoordList *cl, int cap) {
    if (cap < 16) cap = 16;
    cl->data     = malloc((size_t)cap * sizeof(Coord));
    cl->count    = 0;
    cl->capacity = cap;
}

static void cl_push(CoordList *cl, int x, int y, int z) {
    if (cl->count >= cl->capacity) {
        cl->capacity *= 2;
        cl->data = realloc(cl->data, (size_t)cl->capacity * sizeof(Coord));
    }
    cl->data[cl->count++] = (Coord){x, y, z};
}

static void cl_clear(CoordList *cl) { cl->count = 0; }
static void cl_free(CoordList *cl)  { free(cl->data); }

/* Row-major comparison for deterministic output order */
static int coord_cmp(const void *a, const void *b) {
    const Coord *ca = (const Coord *)a;
    const Coord *cb = (const Coord *)b;
    if (ca->x != cb->x) return ca->x - cb->x;
    if (ca->y != cb->y) return ca->y - cb->y;
    return ca->z - cb->z;
}

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

static inline int wrap(int v, int s) {
    int r = v % s;
    return r < 0 ? r + s : r;
}

/* Rule 4555 */
static inline int apply_rule(int alive, int nbrs) {
    return alive ? (nbrs == 4 || nbrs == 5) : (nbrs == 5);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char *argv[]) {
    int size        = 30;
    int generations = 20;
    int seed        = 42;
    int num_threads = 0;

    if (argc > 1) { size = atoi(argv[1]);
                    if (size < 5) { puts("Size too small (min 5)."); return 1; } }
    if (argc > 2) { generations = atoi(argv[2]);
                    if (generations < 1) { puts("Generations must be >= 1."); return 1; } }
    if (argc > 3) { seed = atoi(argv[3]); }
    if (argc > 4) { num_threads = atoi(argv[4]);
                    if (num_threads < 1) { puts("Threads must be >= 1."); return 1; }
                    omp_set_num_threads(num_threads); }

    int actual_threads;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    printf("Initializing a %dx%dx%d universe for %d generations "
           "(Seed: %d, Threads: %d)...\n",
           size, size, size, generations, seed, actual_threads);

    /* --- Seed primordial soup over the entire N³ universe --- */
    CoordList living;
    cl_init(&living, size * size * size / 2 + 64);

    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    for (int k = 0; k < size; k++)
        if (rand() % 2) cl_push(&living, i, j, k);

    printf("Initial live cells: %d\n", living.count);

    /* --- Allocate flat arrays (reused every generation) ----------- */
    int total_cells = size * size * size;
    int ss          = size * size;            /* x-stride */

    char *alive_arr = calloc((size_t)total_cells, sizeof(char));

    /* Per-thread private neighbour-count arrays (no atomics needed) */
    int **local_nbr = malloc((size_t)actual_threads * sizeof(int *));
    for (int t = 0; t < actual_threads; t++)
        local_nbr[t] = calloc((size_t)total_cells, sizeof(int));

    /* Per-thread result lists for parallel rule evaluation */
    CoordList *thr_results = malloc((size_t)actual_threads * sizeof(CoordList));
    for (int t = 0; t < actual_threads; t++)
        cl_init(&thr_results[t], 1024);

    CoordList new_living;
    cl_init(&new_living, living.count + 64);

    /* --- Output file --- */
    FILE *file = fopen("evolution.txt", "w");
    if (!file) { puts("Error opening file!"); return 1; }
    setvbuf(file, NULL, _IOFBF, 1 << 20);

    double t0 = omp_get_wtime();

    /* ============================================================== */
    /*  Simulation — one persistent parallel region for all gens       */
    /* ============================================================== */
    #pragma omp parallel
    {
        int  tid    = omp_get_thread_num();
        int *my_nbr = local_nbr[tid];

        for (int gen = 0; gen < generations; gen++) {

            /* --- 1. Sort + log (single thread) -------------------- */
            #pragma omp single
            {
                qsort(living.data, (size_t)living.count,
                      sizeof(Coord), coord_cmp);

                fprintf(file, "=== Generation %d ===\n", gen);
                for (int i = 0; i < living.count; i++)
                    fprintf(file, "(%d, %d, %d)\n",
                            living.data[i].x,
                            living.data[i].y,
                            living.data[i].z);
                fprintf(file, "\n");
            }
            /* implicit barrier — sorted living visible to all */

            /* --- 2. Parallel scatter ------------------------------
             *
             * Each thread processes its chunk of living cells.
             *   alive_arr : no race (each coordinate is unique).
             *   my_nbr    : thread-private array, no sharing at all.
             * -------------------------------------------------------- */
            #pragma omp for schedule(static)
            for (int i = 0; i < living.count; i++) {
                int x = living.data[i].x;
                int y = living.data[i].y;
                int z = living.data[i].z;

                alive_arr[x * ss + y * size + z] = 1;

                for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    if (!dx && !dy && !dz) continue;
                    int nx = wrap(x + dx, size);
                    int ny = wrap(y + dy, size);
                    int nz = wrap(z + dz, size);
                    my_nbr[nx * ss + ny * size + nz]++;
                }
            }
            /* implicit barrier — all scatter complete */

            /* --- 3. Fused reduce + evaluate + clear ----------------
             *
             * For every grid cell, sum the per-thread neighbour counts,
             * apply Rule 4555, collect survivors/births, and zero the
             * arrays in one pass so they are ready for the next gen.
             * -------------------------------------------------------- */
            cl_clear(&thr_results[tid]);

            #pragma omp for schedule(static)
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++)
                for (int z = 0; z < size; z++) {
                    int idx = x * ss + y * size + z;

                    int is_alive = alive_arr[idx];
                    alive_arr[idx] = 0;

                    int nbrs = 0;
                    for (int t = 0; t < actual_threads; t++) {
                        nbrs += local_nbr[t][idx];
                        local_nbr[t][idx] = 0;
                    }

                    if (nbrs == 0 && !is_alive) continue;

                    if (apply_rule(is_alive, nbrs))
                        cl_push(&thr_results[tid], x, y, z);
                }
            }
            /* implicit barrier — all results collected, arrays cleared */

            /* --- 4. Concatenate + swap (single thread) ------------ */
            #pragma omp single
            {
                int total = 0;
                for (int t = 0; t < actual_threads; t++)
                    total += thr_results[t].count;

                cl_clear(&new_living);
                if (new_living.capacity < total) {
                    new_living.capacity = total + 64;
                    new_living.data = realloc(new_living.data,
                                    (size_t)new_living.capacity * sizeof(Coord));
                }
                for (int t = 0; t < actual_threads; t++) {
                    if (thr_results[t].count > 0) {
                        memcpy(&new_living.data[new_living.count],
                               thr_results[t].data,
                               (size_t)thr_results[t].count * sizeof(Coord));
                        new_living.count += thr_results[t].count;
                    }
                }

                CoordList tmp = living;
                living     = new_living;
                new_living = tmp;

                if (gen % 5 == 0 || gen == generations - 1)
                    printf("  gen %4d : %d live cells\n", gen, living.count);
            }
            /* implicit barrier — new living visible to all */
        }
    }

    double elapsed = omp_get_wtime() - t0;
    printf("Simulation time: %.4f s\n", elapsed);

    fclose(file);
    cl_free(&living);
    cl_free(&new_living);
    for (int t = 0; t < actual_threads; t++) {
        cl_free(&thr_results[t]);
        free(local_nbr[t]);
    }
    free(local_nbr);
    free(thr_results);
    free(alive_arr);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

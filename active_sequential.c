/*
 * 3-D Game of Life 4555 — Active-cell flat-array sequential version
 *
 * Instead of scanning every cell in the N³ universe each generation,
 * this implementation keeps a list of living cells and uses flat arrays
 * to tally neighbour counts only for cells that *could* change state
 * (living cells and their immediate neighbours).
 *
 * Uses flat arrays for alive status and neighbour counts, eliminating
 * all hash-table overhead.  A fused evaluate + clear pass zeros the
 * arrays while producing the next generation, so no separate memset
 * is needed.  Arrays are allocated once and reused across all
 * generations.
 *
 * Complexity per generation: O(26p + N³)  where p = number of living
 * cells.  The N³ term comes from the evaluation scan; for small-to-
 * medium N this is negligible and the constant factor is far smaller
 * than hash-table probing.
 *
 * Algorithm:
 *   1. For every living cell, mark it alive in the flat array and
 *      increment each of its 26 neighbours' counts.
 *   2. Scan the grid: apply rule 4555, collect survivors/births,
 *      and zero the arrays for the next generation — all in one pass.
 *
 * Rule 4555:
 *   - Alive  & 4 or 5 neighbours → survives
 *   - Dead   & exactly 5 neighbours → born
 *   - Otherwise → dead
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Coordinate type                                                    */
/* ------------------------------------------------------------------ */

typedef struct { int x, y, z; } Coord;

/* Wrap a coordinate into [0, s) for toroidal boundary conditions */
static inline int wrap(int v, int s) {
    int r = v % s;
    return r < 0 ? r + s : r;
}

/* ------------------------------------------------------------------ */
/*  Dynamic coordinate list                                            */
/* ------------------------------------------------------------------ */

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

/* Row-major comparison for deterministic, consistent output order */
static int coord_cmp(const void *a, const void *b) {
    const Coord *ca = (const Coord *)a;
    const Coord *cb = (const Coord *)b;
    if (ca->x != cb->x) return ca->x - cb->x;
    if (ca->y != cb->y) return ca->y - cb->y;
    return ca->z - cb->z;
}

/* ------------------------------------------------------------------ */
/*  Rule 4555                                                          */
/* ------------------------------------------------------------------ */

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
    int size        = 30;   /* default grid edge length */
    int generations = 20;   /* default generation count */
    int seed        = 42;   /* default RNG seed         */

    if (argc > 1) {
        size = atoi(argv[1]);
        if (size < 5) {
            printf("Size too small. Please use a size of at least 5.\n");
            return 1;
        }
    }
    if (argc > 2) {
        generations = atoi(argv[2]);
        if (generations < 1) {
            printf("Generations must be at least 1.\n");
            return 1;
        }
    }
    if (argc > 3) {
        seed = atoi(argv[3]);
    }

    printf("Initializing a %dx%dx%d universe for %d generations (Seed: %d)...\n",
           size, size, size, generations, seed);

    /* --- Build initial living-cell list over the ENTIRE N³ universe --- */
    CoordList living;
    cl_init(&living, size * size * size / 2 + 64);

    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                if (rand() % 2)
                    cl_push(&living, i, j, k);

    /* --- Allocate flat arrays (reused every generation) ----------- */
    int total_cells = size * size * size;
    int ss          = size * size;            /* x-stride */

    char *alive_arr = calloc((size_t)total_cells, sizeof(char));
    int  *nbr_count = calloc((size_t)total_cells, sizeof(int));

    /* --- Output file --- */
    FILE *file = fopen("evolution.txt", "w");
    if (!file) {
        printf("Error opening file!\n");
        cl_free(&living);
        return 1;
    }
    setvbuf(file, NULL, _IOFBF, 1 << 20);   /* 1 MiB write buffer */

    clock_t t0 = clock();

    CoordList new_living;
    cl_init(&new_living, 1024);

    /* ============================================================== */
    /*  Simulation loop                                                */
    /* ============================================================== */

    for (int gen = 0; gen < generations; gen++) {

        /* --- 1. Log living cells (sorted for consistent output) ---- */
        qsort(living.data, (size_t)living.count, sizeof(Coord), coord_cmp);

        fprintf(file, "=== Generation %d ===\n", gen);
        for (int i = 0; i < living.count; i++)
            fprintf(file, "(%d, %d, %d)\n",
                    living.data[i].x, living.data[i].y, living.data[i].z);
        fprintf(file, "\n");

        /* --- 2. Mark alive + scatter neighbour counts (one pass) --- */
        for (int i = 0; i < living.count; i++) {
            int x = living.data[i].x;
            int y = living.data[i].y;
            int z = living.data[i].z;

            alive_arr[x * ss + y * size + z] = 1;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = wrap(x + dx, size);
                int ny = wrap(y + dy, size);
                int nz = wrap(z + dz, size);
                nbr_count[nx * ss + ny * size + nz]++;
            }
        }

        /* --- 3. Fused evaluate + clear -----------------------------
         *
         * Scan every grid cell, apply Rule 4555, collect survivors/
         * births, and zero alive_arr + nbr_count in one pass so they
         * are ready for the next generation (no separate memset).
         * ----------------------------------------------------------  */
        cl_clear(&new_living);
        for (int x = 0; x < size; x++)
        for (int y = 0; y < size; y++)
        for (int z = 0; z < size; z++) {
            int idx = x * ss + y * size + z;

            int is_alive = alive_arr[idx];
            alive_arr[idx] = 0;

            int nbrs = nbr_count[idx];
            nbr_count[idx] = 0;

            if (nbrs == 0 && !is_alive) continue;

            if (apply_rule(is_alive, nbrs))
                cl_push(&new_living, x, y, z);
        }

        /* --- 4. Swap living lists ---------------------------------- */
        CoordList tmp = living;
        living     = new_living;
        new_living = tmp;
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Simulation time: %.4f s\n", elapsed);

    fclose(file);
    cl_free(&living);
    cl_free(&new_living);
    free(alive_arr);
    free(nbr_count);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

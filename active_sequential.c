/*
 * 3-D Game of Life 4555 — Active-cell (hash-table) sequential version
 *
 * Instead of scanning every cell in the N³ universe each generation,
 * this implementation keeps a list of living cells and uses a hash map
 * to tally neighbour counts only for cells that *could* change state
 * (living cells and their immediate neighbours).
 *
 * Complexity per generation: O(p)  where p = number of living cells,
 * compared with O(N³) for the brute-force grid scan.
 *
 * Algorithm (Method B, Option 2 from Bays's paper):
 *   1. For every living cell, mark it "alive" in a hash map and add +1
 *      to each of its 26 neighbours' counts.
 *   2. Walk the hash map: apply rule 4555 to every entry.
 *      - Alive  & 4 or 5 neighbours → survives
 *      - Dead   & exactly 5 neighbours → born
 *      - Otherwise → dead
 *   3. Collect survivors/births into the new living-cell list; discard
 *      the hash map.
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
/*  Open-addressing hash map:  (x, y, z) → { alive, nbr_count }      */
/*                                                                     */
/*  Linear probing, automatic 2× resize when load ≥ 50 %.             */
/* ------------------------------------------------------------------ */

typedef struct {
    int           x, y, z;
    unsigned char alive;      /* 1 if the cell is alive this gen   */
    int           nbr_count;  /* number of living neighbours       */
    unsigned char occupied;   /* 1 if slot is in use               */
} Entry;

typedef struct {
    Entry *buckets;
    int    capacity;
    int    count;
} HashMap;

static inline unsigned int hash3(int x, int y, int z) {
    unsigned int h = (unsigned int)x * 73856093u;
    h ^= (unsigned int)y * 19349663u;
    h ^= (unsigned int)z * 83492791u;
    return h;
}

static void hm_init(HashMap *hm, int cap) {
    hm->capacity = cap < 64 ? 64 : cap;
    hm->count    = 0;
    hm->buckets  = calloc((size_t)hm->capacity, sizeof(Entry));
}

static void hm_free(HashMap *hm) { free(hm->buckets); }

/* Low-level insert-or-find (no resize check) */
static Entry *hm_probe(HashMap *hm, int x, int y, int z) {
    unsigned int h = hash3(x, y, z) % (unsigned int)hm->capacity;
    for (;;) {
        Entry *e = &hm->buckets[h];
        if (!e->occupied) {
            e->x = x;  e->y = y;  e->z = z;
            e->alive     = 0;
            e->nbr_count = 0;
            e->occupied  = 1;
            hm->count++;
            return e;
        }
        if (e->x == x && e->y == y && e->z == z)
            return e;
        h = (h + 1) % (unsigned int)hm->capacity;
    }
}

/* Double the capacity and rehash every entry */
static void hm_resize(HashMap *hm) {
    int    old_cap = hm->capacity;
    Entry *old     = hm->buckets;

    hm->capacity *= 2;
    hm->buckets   = calloc((size_t)hm->capacity, sizeof(Entry));
    hm->count     = 0;

    for (int i = 0; i < old_cap; i++) {
        if (old[i].occupied) {
            Entry *e     = hm_probe(hm, old[i].x, old[i].y, old[i].z);
            e->alive     = old[i].alive;
            e->nbr_count = old[i].nbr_count;
        }
    }
    free(old);
}

/* Public lookup — automatically resizes when load factor ≥ 0.5 */
static Entry *hm_get(HashMap *hm, int x, int y, int z) {
    if (hm->count * 2 >= hm->capacity)
        hm_resize(hm);
    return hm_probe(hm, x, y, z);
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

    /* --- Build initial living-cell list (same RNG sequence as grid_sequential) --- */
    CoordList living;
    cl_init(&living, 1024);

    srand((unsigned int)seed);
    int cen = size / 2;
    for (int i = cen - 3; i <= cen + 3; i++)
        for (int j = cen - 3; j <= cen + 3; j++)
            for (int k = cen - 3; k <= cen + 3; k++)
                if (rand() % 2)
                    cl_push(&living, i, j, k);

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

        /* --- 2. Build neighbour-count hash map --------------------- */
        /*
         * For every living cell we:
         *   a) mark it alive in the map, and
         *   b) add +1 to each of its 26 neighbours' counts.
         *
         * After this phase, every occupied entry holds the exact number
         * of living neighbours for that coordinate.  Dead entries whose
         * count reaches exactly 5 may give birth.
         */
        int est = living.count * 60 + 128;   /* generous initial cap */
        HashMap nbr_map;
        hm_init(&nbr_map, est);

        /* a) mark alive */
        for (int i = 0; i < living.count; i++) {
            Entry *e = hm_get(&nbr_map,
                              living.data[i].x,
                              living.data[i].y,
                              living.data[i].z);
            e->alive = 1;
        }

        /* b) scatter +1 to 26 neighbours */
        for (int i = 0; i < living.count; i++) {
            int x = living.data[i].x;
            int y = living.data[i].y;
            int z = living.data[i].z;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        int nx = wrap(x + dx, size);
                        int ny = wrap(y + dy, size);
                        int nz = wrap(z + dz, size);
                        hm_get(&nbr_map, nx, ny, nz)->nbr_count++;
                    }
        }

        /* --- 3. Apply rule, collect survivors and births ----------- */
        cl_clear(&new_living);
        for (int i = 0; i < nbr_map.capacity; i++) {
            Entry *e = &nbr_map.buckets[i];
            if (!e->occupied) continue;
            if (apply_rule(e->alive, e->nbr_count))
                cl_push(&new_living, e->x, e->y, e->z);
        }

        /* --- 4. Swap living lists ---------------------------------- */
        CoordList tmp = living;
        living     = new_living;
        new_living = tmp;
        cl_clear(&new_living);

        hm_free(&nbr_map);
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Simulation time: %.4f s\n", elapsed);

    fclose(file);
    cl_free(&living);
    cl_free(&new_living);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}


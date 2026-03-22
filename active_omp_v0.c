/*
 * 3-D Game of Life – Hash-table version  (Bays 1987, Method B Option 2)
 *
 * Algorithm
 * ---------
 * Maintain two open-addressing hash tables:
 *
 *   alive   – the set of currently living cells
 *   nbr     – for every cell adjacent to ≥1 live cell, its live-
 *              neighbour count (1..26)
 *
 * Per generation:
 *   1. Candidate set = all keys in nbr (every cell near a live cell)
 *      plus any live cell with nbr == 0 (isolated; it will die, but
 *      must not be skipped).
 *   2. Evaluate Rule 4555 on every candidate (read-only phase).
 *   3. Rebuild alive from the evaluation results.
 *   4. Rebuild nbr from the new alive set.
 *
 * Steps 3-4 rebuild from scratch each generation, which is correct and
 * avoids the subtle incremental-update bugs that arise when births/deaths
 * are applied while the neighbour table is still being read.
 *
 * Complexity per generation: O(26p) where p = live-cell count,
 * versus O(N³) for the dense sweep.  Life 4555 residue density is
 * ~0.0005 (Bays 1987, table 3), so for large N this is orders of
 * magnitude fewer cells to visit.
 *
 * Rule 4555 (E_lo=4, E_hi=5, F_lo=5, F_hi=5):
 *   alive → alive  iff  nbrs ∈ {4,5}
 *   dead  → alive  iff  nbrs == 5
 *
 * Boundary: toroidal (wrapping), identical to the dense version.
 *
 * Compile:
 *   gcc -O2 -fopenmp -o grid_hashtable grid_hashtable.c
 *
 * Usage:
 *   ./grid_hashtable  [size]  [generations]  [seed]  [threads]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <stdarg.h>

/* ================================================================== */
/*  Dynamic string buffer for deferred file output                     */
/* ================================================================== */

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

/* ================================================================== */
/*  Packed cell key                                                     */
/*                                                                     */
/*  Coordinates are offset by +1 before packing so that the valid key  */
/*  for (0,0,0) is nonzero, allowing 0 to serve as the empty sentinel. */
/*  Each axis gets 21 bits → max grid size = 2^21-2 = 2,097,150.       */
/* ================================================================== */

#define EMPTY_KEY  UINT64_C(0)
#define COORD_BITS 21
#define COORD_MASK UINT64_C(0x1FFFFF)

static inline uint64_t cell_pack(int x, int y, int z) {
    return (((uint64_t)(unsigned)(x + 1)) << (2 * COORD_BITS))
         | (((uint64_t)(unsigned)(y + 1)) <<      COORD_BITS )
         |  ((uint64_t)(unsigned)(z + 1));
}

static inline void cell_unpack(uint64_t key, int *x, int *y, int *z) {
    *x = (int)((key >> (2 * COORD_BITS)) & COORD_MASK) - 1;
    *y = (int)((key >>      COORD_BITS ) & COORD_MASK) - 1;
    *z = (int)( key                      & COORD_MASK) - 1;
}

static inline int coord_wrap(int v, int s) {
    v %= s;
    return (v < 0) ? v + s : v;
}

static inline uint64_t cell_pack_wrap(int x, int y, int z, int s) {
    return cell_pack(coord_wrap(x, s), coord_wrap(y, s), coord_wrap(z, s));
}

/* Multiplicative Fibonacci hash for 64-bit keys */
static inline size_t ht_hash(uint64_t key, size_t mask) {
    return (size_t)(key * UINT64_C(11400714819323198485)) & mask;
}

/* ================================================================== */
/*  Open-addressing key-only hash set (alive cells)                   */
/* ================================================================== */

typedef struct { uint64_t *keys; size_t cap, count; } KeySet;

static KeySet *ks_new(size_t cap) {
    KeySet *s = malloc(sizeof *s);
    s->cap = cap; s->count = 0;
    s->keys = calloc(cap, sizeof(uint64_t));   /* 0 == EMPTY_KEY */
    return s;
}
static void ks_free(KeySet *s) { free(s->keys); free(s); }

static int  ks_insert(KeySet *s, uint64_t key);   /* forward */
static void ks_grow  (KeySet *s) {
    size_t oc = s->cap; uint64_t *ok = s->keys;
    s->cap *= 2; s->count = 0;
    s->keys = calloc(s->cap, sizeof(uint64_t));
    for (size_t i = 0; i < oc; i++) if (ok[i] != EMPTY_KEY) ks_insert(s, ok[i]);
    free(ok);
}
static int ks_insert(KeySet *s, uint64_t key) {
    if (s->count * 2 >= s->cap) ks_grow(s);
    size_t h = ht_hash(key, s->cap - 1);
    while (s->keys[h] != EMPTY_KEY && s->keys[h] != key)
        h = (h + 1) & (s->cap - 1);
    if (s->keys[h] == EMPTY_KEY) { s->keys[h] = key; s->count++; return 1; }
    return 0;   /* already present */
}
static int ks_contains(const KeySet *s, uint64_t key) {
    size_t h = ht_hash(key, s->cap - 1);
    while (s->keys[h] != EMPTY_KEY) {
        if (s->keys[h] == key) return 1;
        h = (h + 1) & (s->cap - 1);
    }
    return 0;
}
/* Write all keys into a preallocated array; return count */
static size_t ks_snapshot(const KeySet *s, uint64_t *out) {
    size_t n = 0;
    for (size_t i = 0; i < s->cap; i++)
        if (s->keys[i] != EMPTY_KEY) out[n++] = s->keys[i];
    return n;
}

/* ================================================================== */
/*  Open-addressing neighbour-count hash map  (key → int)             */
/* ================================================================== */

typedef struct { uint64_t *keys; int *vals; size_t cap, count; } NbrTable;

static NbrTable *nt_new(size_t cap) {
    NbrTable *t = malloc(sizeof *t);
    t->cap = cap; t->count = 0;
    t->keys = calloc(cap, sizeof(uint64_t));
    t->vals = calloc(cap, sizeof(int));
    return t;
}
static void nt_free(NbrTable *t) { free(t->keys); free(t->vals); free(t); }

static void nt_increment(NbrTable *t, uint64_t key);   /* forward */
static void nt_grow     (NbrTable *t) {
    size_t oc = t->cap; uint64_t *ok = t->keys; int *ov = t->vals;
    t->cap *= 2; t->count = 0;
    t->keys = calloc(t->cap, sizeof(uint64_t));
    t->vals = calloc(t->cap, sizeof(int));
    for (size_t i = 0; i < oc; i++) {
        if (ok[i] == EMPTY_KEY) continue;
        /* Re-insert by opening a slot then writing the saved count */
        size_t h = ht_hash(ok[i], t->cap - 1);
        while (t->keys[h] != EMPTY_KEY) h = (h + 1) & (t->cap - 1);
        t->keys[h] = ok[i]; t->vals[h] = ov[i]; t->count++;
    }
    free(ok); free(ov);
}
static void nt_increment(NbrTable *t, uint64_t key) {
    if (t->count * 2 >= t->cap) nt_grow(t);
    size_t h = ht_hash(key, t->cap - 1);
    while (t->keys[h] != EMPTY_KEY && t->keys[h] != key)
        h = (h + 1) & (t->cap - 1);
    if (t->keys[h] == EMPTY_KEY) { t->keys[h] = key; t->count++; }
    t->vals[h]++;
}
static int nt_get(const NbrTable *t, uint64_t key) {
    size_t h = ht_hash(key, t->cap - 1);
    while (t->keys[h] != EMPTY_KEY) {
        if (t->keys[h] == key) return t->vals[h];
        h = (h + 1) & (t->cap - 1);
    }
    return 0;
}

/* ================================================================== */
/*  Build nbr table from alive set (O(26p))                           */
/* ================================================================== */

static void build_nbr(const KeySet *alive, NbrTable *nbr, int s) {
    for (size_t i = 0; i < alive->cap; i++) {
        if (alive->keys[i] == EMPTY_KEY) continue;
        int x, y, z;
        cell_unpack(alive->keys[i], &x, &y, &z);
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            if (!dx && !dy && !dz) continue;
            nt_increment(nbr, cell_pack_wrap(x+dx, y+dy, z+dz, s));
        }
    }
}

/* ================================================================== */
/*  Growing key buffer (for candidates, births, deaths)               */
/* ================================================================== */

typedef struct { uint64_t *data; size_t len, cap; } Buf;

static void buf_push(Buf *b, uint64_t v) {
    if (b->len == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 256;
        b->data = realloc(b->data, b->cap * sizeof(uint64_t));
    }
    b->data[b->len++] = v;
}

/* ================================================================== */
/*  Comparator for qsort – defined at file scope for Clang compat.    */
/* ================================================================== */

static int cmp64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char *argv[]) {
    int size        = 30;
    int generations = 20;
    int seed        = 42;
    int num_threads = 0;

    if (argc > 1) { size        = atoi(argv[1]);
                    if (size < 5) { puts("Size too small (min 5)."); return 1; } }
    if (argc > 2) { generations = atoi(argv[2]);
                    if (generations < 1) { puts("Generations must be >= 1."); return 1; } }
    if (argc > 3) { seed        = atoi(argv[3]); }
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
    size_t est = (size_t)size * size * size;
    size_t ht_cap = 1;
    while (ht_cap < est * 2) ht_cap *= 2;          /* power-of-2 ≥ 2·N³ */
    KeySet  *alive = ks_new(ht_cap);
    NbrTable *nbr  = nt_new(ht_cap * 4);

    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    for (int k = 0; k < size; k++)
        if (rand() % 2) ks_insert(alive, cell_pack(i, j, k));

    build_nbr(alive, nbr, size);
    printf("Initial live cells: %zu\n", alive->count);

    /* --- In-memory output buffer (written to file at the end) --- */
    StrBuf out_buf;
    sb_init(&out_buf, alive->count * 30 * (size_t)generations);

    /* Scratch arrays (reused each generation) */
    Buf       cands      = {0};
    int      *next_state = NULL;
    size_t    next_state_cap = 0;
    uint64_t *snap       = NULL;
    size_t    snap_cap   = 0;

    double t0 = omp_get_wtime();

    /* ============================================================== */
    /*  Simulation loop                                                */
    /* ============================================================== */
    for (int gen = 0; gen < generations; gen++) {

        /* --- 1. Buffer live cells (sorted for deterministic output) --- */
        sb_printf(&out_buf, "=== Generation %d ===\n", gen);

        if (alive->count > snap_cap) {
            snap_cap = alive->count * 2 + 64;
            snap = realloc(snap, snap_cap * sizeof(uint64_t));
        }
        size_t n_alive = ks_snapshot(alive, snap);

        /* sort keys for deterministic output order */
        qsort(snap, n_alive, sizeof(uint64_t), cmp64);

        for (size_t i = 0; i < n_alive; i++) {
            int cx, cy, cz;
            cell_unpack(snap[i], &cx, &cy, &cz);
            sb_printf(&out_buf, "(%d, %d, %d)\n", cx, cy, cz);
        }
        sb_printf(&out_buf, "\n");

        /* --- 2. Collect candidates --------------------------------
         *
         * Candidates = all keys in nbr (cells adjacent to ≥1 live cell)
         * plus any live cell with no neighbours (isolated; will die).
         * ---------------------------------------------------------*/
        cands.len = 0;

        /* Every cell in nbr is a candidate */
        for (size_t i = 0; i < nbr->cap; i++)
            if (nbr->keys[i] != EMPTY_KEY)
                buf_push(&cands, nbr->keys[i]);

        /* Isolated live cells (alive but nbr count == 0) */
        for (size_t i = 0; i < alive->cap; i++) {
            if (alive->keys[i] == EMPTY_KEY) continue;
            if (nt_get(nbr, alive->keys[i]) == 0)
                buf_push(&cands, alive->keys[i]);
        }
        size_t n_cand = cands.len;

        /* --- 3. Evaluate Rule 4555 in parallel (read-only phase) -- */
        if (n_cand > next_state_cap) {
            next_state_cap = n_cand * 2 + 64;
            next_state = realloc(next_state, next_state_cap * sizeof(int));
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_cand; i++) {
            uint64_t key  = cands.data[i];
            int is_alive  = ks_contains(alive, key);
            int n         = nt_get(nbr, key);
            if (is_alive)
                next_state[i] = (n == 4 || n == 5) ? 1 : 0;
            else
                next_state[i] = (n == 5) ? 1 : 0;
        }

        /* --- 4. Rebuild alive set from evaluation results ----------
         *
         * We free and recreate alive and nbr from scratch.  This is
         * O(26p) — the same cost as the incremental approach — but
         * avoids the correctness hazard of mutating nbr while reads
         * are still in flight from the candidate evaluation above.
         * ---------------------------------------------------------*/
        ks_free(alive);  alive = ks_new(ht_cap);
        nt_free(nbr);    nbr   = nt_new(ht_cap * 4);

        for (size_t i = 0; i < n_cand; i++)
            if (next_state[i])
                ks_insert(alive, cands.data[i]);

        build_nbr(alive, nbr, size);

        if (gen % 5 == 0 || gen == generations - 1)
            printf("  gen %4d : %zu live cells\n", gen, alive->count);
    }

    double elapsed = omp_get_wtime() - t0;
    printf("Simulation time: %.4f s\n", elapsed);

    /* --- Write all output to file at the very end --- */
    FILE *file = fopen("evolution.txt", "w");
    if (!file) { puts("Error opening file!"); return 1; }
    fwrite(out_buf.data, 1, out_buf.len, file);
    fclose(file);

    sb_free(&out_buf);
    ks_free(alive);
    nt_free(nbr);
    free(cands.data);
    free(next_state);
    free(snap);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

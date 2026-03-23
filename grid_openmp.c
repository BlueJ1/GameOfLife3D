// grid_omp.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdarg.h>

// String buffer for outputting evolutionary history
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StrBuf;

static void init_sb(StrBuf *sb, size_t cap) {
    if (cap < 4096) cap = 4096;
    sb->data = malloc(cap);
    sb->len  = 0;
    sb->cap  = cap;
}

static void printf_sb(StrBuf *sb, const char *fmt, ...) {
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

static void free_sb(StrBuf *sb) { free(sb->data); }

// 3-D → 1-D index
static inline int idx(int x, int y, int z, int size_squared, int size) {
    return x * size_squared + y * size + z;
}

// Count neighbours in interior
static inline int count_interior(const unsigned char *restrict g,
                                 int x, int y, int z,
                                 int size_squared, int size) {
    int c = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int bx = (x + dx) * size_squared;
        for (int dy = -1; dy <= 1; dy++) {
            int bxy = bx + (y + dy) * size;
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                c += g[bxy + (z + dz)];
            }
        }
    }
    return c;
}

// Count neighbours on boundary
static inline int count_boundary(const unsigned char *restrict g,
                                 int x, int y, int z,
                                 int size_squared, int s) {
    int c = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int bx = ((x + dx + s) % s) * size_squared;
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


int main(int argc, char *argv[]) {
    int size = 30;
    int generations = 20;
    int seed = 42;
    int num_threads = 0;

    // CLI arguments: size generations seed threads
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

    printf("Initializing a %dx%dx%d universe for %d generations (Seed: %d, Threads: %d)...\n",
           size, size, size, generations, seed, num_threads);

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

    /* --- In-memory output buffer (written to file at the end) --- */
    StrBuf out_buf;
    init_sb(&out_buf, (size_t)total_size * 15 * (size_t)generations);

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

        /* --- 1. Buffer living cells (file order must be deterministic) --- */
        printf_sb(&out_buf, "=== Generation %d ===\n", gen);
        for (int i = 0; i < total_size; i++) {
            if (grid[i]) {
                int x = i / size_squared;
                int r = i % size_squared;
                printf_sb(&out_buf, "(%d, %d, %d)\n", x, r / size, r % size);
            }
        }

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

        unsigned char *tmp = grid;
        grid      = next_grid;
        next_grid = tmp;
        memset(next_grid, 0, (size_t)total_size);

        printf_sb(&out_buf, "\n");
    }

    double t1 = omp_get_wtime();
    printf("Simulation time: %.4f s\n", t1 - t0);

    FILE *file = fopen("evolution.txt", "w");
    if (!file) {
        puts("Error opening file!");
        free(grid);
        free(next_grid);
        free_sb(&out_buf);
        return 1;
    }
    fwrite(out_buf.data, 1, out_buf.len, file);
    fclose(file);

    free_sb(&out_buf);
    free(grid);
    free(next_grid);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

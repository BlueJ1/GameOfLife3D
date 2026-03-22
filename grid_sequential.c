#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

// apply rule 4555 (survive on 4 or 5 neighbours, birth on 5)
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

    // CLI arguments: size generations seed
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

    const int total_cells = size * size * size;
    const int size_squared = size * size;   /* precomputed size^2 */

    unsigned char *grid = calloc(total_cells, 1);
    unsigned char *next_grid = calloc(total_cells, 1);
    if (!grid || !next_grid) {
        printf("Memory allocation failed. Grid size might be too large.\n");
        return 1;
    }

    StrBuf out_buf;
    init_sb(&out_buf, (size_t)total_cells * 15 * (size_t)generations);

    /* Seed the "primordial soup" over the entire N³ universe */
    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                grid[idx(i, j, k, size_squared, size)] = (unsigned char)(rand() % 2);

    clock_t t0 = clock();

    for (int gen = 0; gen < generations; gen++) {
        printf_sb(&out_buf, "=== Generation %d ===\n", gen);
        for (int i = 0; i < total_cells; i++) {
            if (grid[i]) {
                int x = i / size_squared;
                int r = i % size_squared;
                printf_sb(&out_buf, "(%d, %d, %d)\n", x, r / size, r % size);
            }
        }

        for (int i = 0; i < total_cells; i++) {
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
        grid = next_grid;
        next_grid = tmp;
        memset(next_grid, 0, (size_t)total_cells);

        printf_sb(&out_buf, "\n");
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Simulation time: %.4f s\n", elapsed);

    FILE *file = fopen("evolution.txt", "w");
    if (!file) {
        printf("Error opening file!\n");
        free_sb(&out_buf);
        free(grid);
        free(next_grid);
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

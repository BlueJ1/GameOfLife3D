#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

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

typedef struct { int x, y, z; } Coord;

// Wrap coordinate into [0, size) for boundary
static inline int wrap(int v, int size) {
    int r = v % size;
    return r < 0 ? r + size : r;
}

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

static inline unsigned char apply_rule(unsigned char alive, int nbrs) {
    if (alive) {
        return (nbrs == 4 || nbrs == 5) ? 1 : 0;
    } else {
        return (nbrs == 5) ? 1 : 0;
    }
}

int main(int argc, char *argv[]) {
    int size = 30;
    int generations = 20;
    int seed = 42;

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

    CoordList living;
    cl_init(&living, size * size * size / 2 + 64);

    srand((unsigned int)seed);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                if (rand() % 2)
                    cl_push(&living, i, j, k);

    int total_cells = size * size * size;
    int size_squared = size * size;            /* x-stride */

    char *alive_arr = calloc((size_t)total_cells, sizeof(char));
    int  *nbr_count = calloc((size_t)total_cells, sizeof(int));

    StrBuf out_buf;
    init_sb(&out_buf, (size_t)living.count * 30 * (size_t)generations);

    clock_t t0 = clock();

    CoordList new_living;
    cl_init(&new_living, 1024);

    for (int gen = 0; gen < generations; gen++) {
        printf_sb(&out_buf, "=== Generation %d ===\n", gen);
        for (int i = 0; i < living.count; i++)
            printf_sb(&out_buf, "(%d, %d, %d)\n",
                    living.data[i].x, living.data[i].y, living.data[i].z);
        printf_sb(&out_buf, "\n");

        for (int i = 0; i < living.count; i++) {
            int x = living.data[i].x;
            int y = living.data[i].y;
            int z = living.data[i].z;

            alive_arr[x * size_squared + y * size + z] = 1;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = wrap(x + dx, size);
                int ny = wrap(y + dy, size);
                int nz = wrap(z + dz, size);
                nbr_count[nx * size_squared + ny * size + nz]++;
            }
        }

        cl_clear(&new_living);
        for (int x = 0; x < size; x++)
        for (int y = 0; y < size; y++)
        for (int z = 0; z < size; z++) {
            int idx = x * size_squared + y * size + z;

            int is_alive = alive_arr[idx];
            alive_arr[idx] = 0;

            int nbrs = nbr_count[idx];
            nbr_count[idx] = 0;

            if (nbrs == 0 && !is_alive) continue;

            if (apply_rule(is_alive, nbrs))
                cl_push(&new_living, x, y, z);
        }

        CoordList tmp = living;
        living = new_living;
        new_living = tmp;
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Simulation time: %.4f s\n", elapsed);

    FILE *file = fopen("evolution.txt", "w");
    if (!file) {
        printf("Error opening file!\n");
        free_sb(&out_buf);
        cl_free(&living);
        cl_free(&new_living);
        free(alive_arr);
        free(nbr_count);
        return 1;
    }
    fwrite(out_buf.data, 1, out_buf.len, file);
    fclose(file);

    free_sb(&out_buf);
    cl_free(&living);
    cl_free(&new_living);
    free(alive_arr);
    free(nbr_count);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

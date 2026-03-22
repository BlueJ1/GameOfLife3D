#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdarg.h>

typedef struct { int x, y, z; } Coord;

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
    cl->data[cl->count] = (Coord){x, y, z};
    cl->count++;
}

static void cl_clear(CoordList *cl) { cl->count = 0; }
static void cl_free(CoordList *cl)  { free(cl->data); }

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



static inline int wrap(int v, int size) {
    int r = v % size;
    return r < 0 ? r + size : r;
}

/* Rule 4555 */
static inline int apply_rule(int alive, int nbrs) {
    return alive ? (nbrs == 4 || nbrs == 5) : (nbrs == 5);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char *argv[]) {
    int size = 30;
    int generations = 20;
    int seed = 42;
    int num_threads = 0;

    if (argc > 1) { size = atoi(argv[1]);
                    if (size < 5) { puts("Size too small (min 5)."); return 1; } }
    if (argc > 2) { generations = atoi(argv[2]);
                    if (generations < 1) { puts("Generations must be >= 1."); return 1; } }
    if (argc > 3) { seed = atoi(argv[3]); }
    if (argc > 4) { num_threads = atoi(argv[4]);
                    if (num_threads < 1) { puts("Threads must be >= 1."); return 1; }
                    omp_set_num_threads(num_threads); }

    printf("Initializing a %dx%dx%d universe for %d generations "
           "(Seed: %d, Threads: %d)...\n",
           size, size, size, generations, seed, num_threads);

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
    int **local_nbr = malloc((size_t)num_threads * sizeof(int *));
    for (int t = 0; t < num_threads; t++)
        local_nbr[t] = calloc((size_t)total_cells, sizeof(int));

    /* Per-thread result lists for parallel rule evaluation */
    CoordList *thr_results = malloc((size_t)num_threads * sizeof(CoordList));
    for (int t = 0; t < num_threads; t++)
        cl_init(&thr_results[t], 1024);

    CoordList new_living;
    cl_init(&new_living, living.count + 64);

    StrBuf out_buf;
    init_sb(&out_buf, (size_t)living.count * 30 * (size_t)generations);

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int  tid    = omp_get_thread_num();
        int *my_nbr = local_nbr[tid];

        for (int gen = 0; gen < generations; gen++) {
            #pragma omp single
            {

                printf_sb(&out_buf, "=== Generation %d ===\n", gen);
                for (int i = 0; i < living.count; i++)
                    printf_sb(&out_buf, "(%d, %d, %d)\n",
                            living.data[i].x,
                            living.data[i].y,
                            living.data[i].z);
                printf_sb(&out_buf, "\n");
            }

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

            cl_clear(&thr_results[tid]);

            #pragma omp for schedule(static)
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++)
                for (int z = 0; z < size; z++) {
                    int idx = x * ss + y * size + z;

                    int is_alive = alive_arr[idx];
                    alive_arr[idx] = 0;

                    int nbrs = 0;
                    for (int t = 0; t < num_threads; t++) {
                        nbrs += local_nbr[t][idx];
                        local_nbr[t][idx] = 0;
                    }

                    if (nbrs == 0 && !is_alive) continue;

                    if (apply_rule(is_alive, nbrs))
                        cl_push(&thr_results[tid], x, y, z);
                }
            }

            #pragma omp single
            {
                int total = 0;
                for (int t = 0; t < num_threads; t++)
                    total += thr_results[t].count;

                cl_clear(&new_living);
                if (new_living.capacity < total) {
                    new_living.capacity = total + 64;
                    new_living.data = realloc(new_living.data,
                                    (size_t)new_living.capacity * sizeof(Coord));
                }
                for (int t = 0; t < num_threads; t++) {
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
            }
        }
    }

    double elapsed = omp_get_wtime() - t0;
    printf("Simulation time: %.4f s\n", elapsed);

    FILE *file = fopen("evolution.txt", "w");
    if (!file) { puts("Error opening file!"); return 1; }
    fwrite(out_buf.data, 1, out_buf.len, file);
    fclose(file);

    free_sb(&out_buf);
    cl_free(&living);
    cl_free(&new_living);
    for (int t = 0; t < num_threads; t++) {
        cl_free(&thr_results[t]);
        free(local_nbr[t]);
    }
    free(local_nbr);
    free(thr_results);
    free(alive_arr);

    printf("Simulation complete. Results saved to evolution.txt\n");
    return 0;
}

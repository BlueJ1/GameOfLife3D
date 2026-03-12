CC = gcc
CFLAGS = -O3 -Wall
LDLIBS = -lm
LIBOMP_PREFIX = $(shell brew --prefix libomp)
OMPFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp

all: gs as ao ao2 go go2
gs: grid_sequential.c
	$(CC) $(CFLAGS) -o gs grid_sequential.c $(LDLIBS)
as: active_sequential.c
	$(CC) $(CFLAGS) -o as active_sequential.c $(LDLIBS)
ao: active_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o ao active_omp.c $(LDLIBS)
ao2: active_omp_v0.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o ao2 active_omp_v0.c $(LDLIBS)
go: grid_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o go grid_openmp.c $(LDLIBS)
go2: grid_openmp_v0.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o go2 grid_openmp_v0.c $(LDLIBS)
clean:
	rm -f gs as ao ao2 go go2

CC = gcc
CFLAGS = -O3 -Wall
LDLIBS = -lm
LIBOMP_PREFIX = $(shell brew --prefix libomp)
OMPFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp

all: gs go
gs: grid_sequential.c
	$(CC) $(CFLAGS) -o gs grid_sequential.c $(LDLIBS)
go: grid_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o go grid_openmp.c $(LDLIBS)
go2: grid_openmp_v0.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o go2 grid_openmp_v0.c $(LDLIBS)
clean:
	rm -f gs go

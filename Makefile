CC = gcc
CFLAGS = -O3 -Wall -ffast-math -march=native
LDLIBS = -lm
OMPFLAGS = -fopenmp

all: gs as ao go
gs: grid_sequential.c
	$(CC) $(CFLAGS) -o gs grid_sequential.c $(LDLIBS)
as: active_sequential.c
	$(CC) $(CFLAGS) -o as active_sequential.c $(LDLIBS)
ao: active_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o ao active_omp.c $(LDLIBS)
go: grid_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o go grid_openmp.c $(LDLIBS)
clean:
	rm -f gs as ao go

CC = /usr/bin/gcc 
CFLAGS = -g -fopenmp, -pthread, -O3

all: Quicksort
	# generate package
	-tar -cvf ${USER}-handin.tar Quicksort.cpp Report.pdf Makefile

Quicksort: Quicksort.cpp
	$(CC) $(CFLAGS) -o Quicksort Quicksort.cpp

clean:
	rm -rf *.o
	rm -f matrixmult
	rm -f cudamatrixmult
	rm -f *.tar
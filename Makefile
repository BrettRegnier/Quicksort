CC = /usr/bin/g++
CFLAGS = -g -fopenmp -pthread

all: QuicksortO3 Quicksort
	# generate package
	-tar -cvf ${USER}-handin.tar Quicksort.cpp Report.pdf Makefile

QuicksortO3: Quicksort.cpp
	$(CC) $(CFLAGS) -O3 -o QuicksortO3 Quicksort.cpp
	
Quicksort: Quicksort.cpp
	$(CC) $(CFLAGS) -o Quicksort Quicksort.cpp

clean:
	rm -rf *.o
	rm -f Quicksort
	rm -f QuicksortO3
	rm -f *.tar
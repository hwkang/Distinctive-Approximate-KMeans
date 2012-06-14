CXX=g++
CXXFLAGS=-O2 -Wall -I/usr/local/include/flann
LIBS=-lflann -pthread

BIN=DAKMeans

all: ${BIN}

.cc.o:
	@${CXX} -c ${CXXFLAGS} $<

DAKMeans.o: DAKMeans.cpp util.h

DAKMeans: DAKMeans.o
	${CXX} -o $@ $^ ${LIBS}

clean:
	rm -f *.o ${BIN}

.PHONY: all clean

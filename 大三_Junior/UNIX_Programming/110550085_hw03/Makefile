CXX = g++
CFLAGS = -std=c++17 -I. -Wall -g 
LIBS = -lcapstone


all: sdb.cpp
	$(CXX) sdb.cpp $(CFLAGS) -o sdb $(LIBS)

clean:
	rm sdb
	
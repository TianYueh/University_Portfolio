CC = gcc
CFLAGS = -Wall -Wextra -fPIC
LDFLAGS = -shared

all: libsolve.so libsolver.so libmaze.so maze

libmaze.so: libmaze_dummy.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< 

libsolver.so: libsolver.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

libsolve.so: libsolve.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

maze: maze.c libmaze.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lmaze 

clean:
	rm -f libsolver.so libmaze.so maze

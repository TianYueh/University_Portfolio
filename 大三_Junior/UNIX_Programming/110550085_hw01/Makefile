CC = gcc
CFLAGS = -Wall -Wextra -fPIC
LDFLAGS = -shared

all: logger.so logger

logger.so: logger_so.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

logger: logger.c logger.so
	$(CC) $(CFLAGS) -o $@ $< -L. -ldl

clean:
	rm -f logger.so logger

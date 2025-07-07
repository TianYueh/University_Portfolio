#ifndef LOGGER_H
#define LOGGER_H

#include <dlfcn.h>
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <errno.h>
#include <fnmatch.h>
#include <netinet/in.h>

static FILE *(*orig_fopen)(const char *, const char *) = NULL;
static size_t (*orig_fread)(void *, size_t, size_t, FILE *) = NULL;
static size_t (*orig_fwrite)(const void *, size_t, size_t, FILE *) = NULL;
static int (*orig_connect)(int, const struct sockaddr *, socklen_t) = NULL;
static int (*orig_getaddrinfo)(const char *, const char *, const struct addrinfo *, struct addrinfo **) = NULL;
static int (*orig_system)(const char *) = NULL;
static void* (*orig_dlsym)(void*, const char*) = NULL;

/*
extern "C"
{
    FILE *fopen(const char *pathname, const char *mode);
    size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
    size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
    int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
    int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res);
    int system(const char *command);
}
*/


#endif
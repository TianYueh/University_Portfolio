// Define the maximum length for a blacklist entry

#define MAX_BLACKLIST_ENTRY_LENGTH 256

// Global arrays to store blacklisted file names for different functions

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
#include <netdb.h>

static FILE* (*orig_fopen)(const char*, const char*) = NULL;
static size_t(*orig_fread)(void*, size_t, size_t, FILE*) = NULL;
static size_t(*orig_fwrite)(const void*, size_t, size_t, FILE*) = NULL;
static int (*orig_connect)(int, const struct sockaddr*, socklen_t) = NULL;
static int (*orig_getaddrinfo)(const char*, const char*, const struct addrinfo*, struct addrinfo**) = NULL;
static int (*orig_system)(const char*) = NULL;
static void* (*orig_dlsym)(void*, const char*) = NULL;


const char* open_blacklist[MAX_BLACKLIST_ENTRY_LENGTH];
const char* read_blacklist[MAX_BLACKLIST_ENTRY_LENGTH];
const char* write_blacklist[MAX_BLACKLIST_ENTRY_LENGTH];
const char* connect_blacklist[MAX_BLACKLIST_ENTRY_LENGTH];
const char* getaddrinfo_blacklist[MAX_BLACKLIST_ENTRY_LENGTH];
size_t open_blacklist_size = 0;
size_t read_blacklist_size = 0;
size_t write_blacklist_size = 0;
size_t connect_blacklist_size = 0;
size_t getaddrinfo_blacklist_size = 0;

char log_filename[256] = ""; // Buffer to store log file name
FILE* log_file = NULL;



void* get_orig_func(const char* func_name) {
    void* handle = dlopen("libc.so.6", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error in `dlopen`: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    void* ret_func = dlsym(handle, func_name);
    if (!ret_func) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    return ret_func;
}

// Function to load blacklist entries from config file
void load_blacklist_entries(const char* filename, const char* section_start, const char* section_end, const char** blacklist, size_t* blacklist_size) {
    //FILE* file = fopen(filename, "r");

    //static FILE *(*orig_fopen)(const char *, const char *) = NULL;
    if (!orig_fopen) {
        orig_fopen = (FILE * (*)(const char*, const char*))get_orig_func("fopen");
    }

    FILE* file = orig_fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening config file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_BLACKLIST_ENTRY_LENGTH];
    int inside_section = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        // Remove newline character
        line[strcspn(line, "\n")] = '\0';

        if (strcmp(line, section_start) == 0) {
            inside_section = 1;
            continue;
        }
        else if (strcmp(line, section_end) == 0) {
            inside_section = 0;
            break;
        }

        if (inside_section) {
            // Add the entry to the blacklist
            blacklist[(*blacklist_size)++] = strdup(line);
        }
    }

    fclose(file);
}

FILE* fopen(const char* pathname, const char* mode) {
    // Function pointers for the original fopen and dlsym

    //fprintf(stderr, "fopen\n");
    // Load the original functions on first call
    if (!orig_fopen) {
        orig_fopen = (FILE * (*)(const char* pathname, const char* mode))get_orig_func(__func__);

        // Load blacklist entries from config file

    }
    load_blacklist_entries("config.txt", "BEGIN open-blacklist", "END open-blacklist", open_blacklist, &open_blacklist_size);
    size_t i;
    //int x = 0x0;

    // Extract the filename from the pathname
    const char* filename_start = strrchr(pathname, '/');
    const char* filename = (filename_start != NULL) ? filename_start + 1 : pathname;

    // Set log_filename to the extracted filename
    strncpy(log_filename, filename, sizeof(log_filename) - 1);
    log_filename[sizeof(log_filename) - 1] = '\0'; // Ensure null-termination

    //printf("log_filename: %s\n", log_filename);

    for (i = 0; i < open_blacklist_size; ++i) {
        if (fnmatch(open_blacklist[i], pathname, FNM_PATHNAME) == 0) {
            errno = EACCES;
            FILE* ret = 0x0;
            fprintf(stderr, "[logger] fopen(\"%s\", \"%s\") = 0x0\n", pathname, mode);
            return NULL;
        }
    }

    FILE* ret = orig_fopen(pathname, mode);
    fprintf(stderr, "[logger] fopen(\"%s\", \"%s\") = % p\n", pathname, mode, ret);

    // Check if the filename is in any of the blacklists

    return ret;
}







size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    if (!orig_fread) {
        orig_fread = (size_t(*)(void*, size_t, size_t, FILE*))get_orig_func(__func__);
    }

    load_blacklist_entries("config.txt", "BEGIN read-blacklist", "END read-blacklist", read_blacklist, &read_blacklist_size);

    // Calculate the total size to be read
    size_t total_size = size * nmemb;

    // Allocate a temporary buffer to hold the data read from the stream
    void* temp_buf = malloc(total_size);
    if (!temp_buf) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    pid_t pid = getpid();
    //const char* filename_start = strrchr(stream->filename, '/');
    //const char* filename = (filename_start != NULL) ? filename_start + 1 : stream->filename;
    char read_filename[256];
    sprintf(read_filename, "%d--%s--read.log", pid, log_filename);

    //log_file = fopen(read_filename, "a");
    log_file = orig_fopen(read_filename, "a");
    if (!log_file) {
        fprintf(stderr, "Error opening log file: %s\n", read_filename);
        exit(EXIT_FAILURE);
    }

    // Perform the actual read using the original fread function
    size_t bytes_read = orig_fread(temp_buf, size, nmemb, stream);

    // Check if there was any data read
    if (bytes_read > 0) {
        // Convert the data buffer to a char pointer for keyword detection
        char* data = (char*)temp_buf;

        // Iterate through the data read and check for keywords
        size_t i;
        for (i = 0; i < bytes_read; ++i) {
            size_t j;
            for (j = 0; j < read_blacklist_size; ++j) {
                size_t keyword_len = strlen(read_blacklist[j]);
                if (i + keyword_len <= bytes_read && memcmp(data + i, read_blacklist[j], keyword_len) == 0) {
                    // Keyword found, block access and log
                    free(temp_buf);  // Free the temporary buffer
                    errno = EACCES;
                    fprintf(stderr, "[logger] fread(\"%p\", %zu, %zu, %p) = %zu\n", ptr, size, nmemb, stream, 0);
                    fprintf(log_file, "[logger] fread(\"%p\", %zu, %zu, %p) = %zu\n", ptr, size, nmemb, stream, 0);
                    return 0;
                }
            }
        }

        // Copy the data back to the original buffer if no keyword was found
        memcpy(ptr, temp_buf, bytes_read);
    }

    free(temp_buf);  // Free the temporary buffer

    // Log the fread operation
    fprintf(stderr, "[logger] fread(\"%p\", %zu, %zu, %p) = %zu\n", ptr, size, nmemb, stream, bytes_read);
    fprintf(log_file, "[logger] fread(\"%p\", %zu, %zu, %p) = %zu\n", ptr, size, nmemb, stream, bytes_read);
    return bytes_read;

}


size_t fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
    if (!orig_fwrite) {
        orig_fwrite = (size_t(*)(const void*, size_t, size_t, FILE*))get_orig_func(__func__);
    }

    load_blacklist_entries("config.txt", "BEGIN write-blacklist", "END write-blacklist", write_blacklist, &write_blacklist_size);

    pid_t pid = getpid();
    //const char* filename_start = strrchr(stream->filename, '/');
    //const char* filename = (filename_start != NULL) ? filename_start + 1 : stream->filename;
    char write_filename[256];

    sprintf(write_filename, "%d--%s--write.log", pid, log_filename);
    //printf("write_name: %s\n", write_filename);

    log_file = orig_fopen(write_filename, "a");
    if (!log_file) {
        fprintf(stderr, "Error opening log file: %s\n", write_filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < write_blacklist_size; i++) {
        if (fnmatch(write_blacklist[i], ptr, FNM_PATHNAME) == 0) {
            errno = EACCES;
            return 0;
        }
    }

    size_t ret = orig_fwrite(ptr, size, nmemb, stream);

    // Log the fwrite operation
    char* data = (char*)ptr;
    char write_data[256] = { '\0' };
    int k = 0;
    for(int i = 0; i < strlen(data); i++) {
		//write_data[i] = data[i];
        //printf("data[i]: %c\n", data[i]);
        
        if(data[i] == '\n'){
			write_data[k] = '\\';
			write_data[k+1] = 'n';
            //i += 1;
            k += 1;
		}
		else {
			write_data[k] = data[i];
		}
        k += 1;
	}
    fprintf(stderr, "[logger] fwrite(\"%s\", %zu, %zu, %p) = %zu\n", write_data, size, nmemb, stream, ret);
    fprintf(log_file, "[logger] fwrite(\"%s\", %zu, %zu, %p) = %zu\n", write_data, size, nmemb, stream, ret);
    

    return ret;
}














int connect(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
    if (!orig_connect) {
        orig_connect = (int (*)(int, const struct sockaddr*, socklen_t))get_orig_func(__func__);
    }

    load_blacklist_entries("config.txt", "BEGIN connect-blacklist", "END connect-blacklist", connect_blacklist, &connect_blacklist_size);

    // Check if the address family is AF_INET (IPv4) or AF_INET6 (IPv6)
    if (addr->sa_family == AF_INET) {
        struct sockaddr_in* addr_in = (struct sockaddr_in*)addr;
        char ip_buffer[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip_buffer, INET_ADDRSTRLEN);

        // Iterate through the blacklist and compare against the IP address
        size_t i;
        for (i = 0; i < connect_blacklist_size; ++i) {
            if (strcmp(ip_buffer, connect_blacklist[i]) == 0) {
                // IP address found in blacklist, block access and log
                errno = ECONNREFUSED;
                fprintf(stderr, "[logger] connect(%d, \"%s\", %d) = -1\n", sockfd, ip_buffer, addrlen);
                return -1;
            }
        }
        // No IP address found in blacklist, proceed with the original connect function
        int ret = orig_connect(sockfd, addr, addrlen);

        fprintf(stderr, "[logger] connect(%d, \"%s\", %d) = %d\n", sockfd, ip_buffer, addrlen, ret);
        return ret;
    }
    else if (addr->sa_family == AF_INET6) {
        struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)addr;
        char ip_buffer[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &(addr_in6->sin6_addr), ip_buffer, INET6_ADDRSTRLEN);

        // Iterate through the blacklist and compare against the IP address
        size_t i;
        for (i = 0; i < connect_blacklist_size; ++i) {
            if (strcmp(ip_buffer, connect_blacklist[i]) == 0) {
                // IP address found in blacklist, block access and log
                errno = ECONNREFUSED;
                fprintf(stderr, "[logger] connect(%d, \"%s\", %d) = -1\n", sockfd, ip_buffer, addrlen);
                return -1;
            }
        }
        // No IP address found in blacklist, proceed with the original connect function
        int ret = orig_connect(sockfd, addr, addrlen);

        fprintf(stderr, "[logger] connect(%d, \"%s\", %d) = %d\n", sockfd, ip_buffer, addrlen, ret);
        return ret;
    }

    

    
}


int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res) {
    if (!orig_getaddrinfo) {
        orig_getaddrinfo = (int (*)(const char *, const char *, const struct addrinfo *, struct addrinfo **))get_orig_func(__func__);
    }

    // Implement your logging and blocking logic here

    load_blacklist_entries("config.txt", "BEGIN getaddrinfo-blacklist", "END getaddrinfo-blacklist", getaddrinfo_blacklist, &getaddrinfo_blacklist_size);


    for(int i= 0; i < getaddrinfo_blacklist_size; i++) {
        if(fnmatch(getaddrinfo_blacklist[i], node, FNM_PATHNAME) == 0) {
            fprintf(stderr, "[logger] getaddrinfo(\"%s\", %s, %p, %p) = %d\n", node, service, hints, res, EAI_NONAME);
			return EAI_NONAME;
		}
    }

    int ret = orig_getaddrinfo(node, service, hints, res);

    fprintf(stderr, "[logger] getaddrinfo(\"%s\", %s, %p, %p) = %d\n", node, service, hints, res, ret);
    //fprintf(stderr, "IPv4: %s\n", inet_ntoa(((struct sockaddr_in*)((*res)->ai_addr))->sin_addr));
    //fprintf(stderr, "IPv6: %s\n", inet_ntop(AF_INET6, &((struct sockaddr_in6*)((*res)->ai_addr))->sin6_addr, (char*)malloc(INET6_ADDRSTRLEN), INET6_ADDRSTRLEN));


    return ret;
}

int system(const char *command) {
    if (!orig_system) {
        orig_system = (int* (*)(const char*))get_orig_func(__func__);
    }

    load_blacklist_entries("config.txt", "BEGIN system-blacklist", "END system-blacklist", getaddrinfo_blacklist, &getaddrinfo_blacklist_size);

    // Implement your logging and blocking logic here
    int ret = orig_system(command);

    fprintf(stderr, "[logger] system(%s)\n", command);


    return ret;
}


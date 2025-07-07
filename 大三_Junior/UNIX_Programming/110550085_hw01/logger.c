#include <stdio.h>
#include <unistd.h>
#include <linux/limits.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[])
{
    char ch;
    char output_file[PATH_MAX] = {};
    char so_path[PATH_MAX] = "./logger.so";

    if (argc == 1)
    {
        printf("no command given.\n\n");
        return 0;
    }

    int config_arg = 0;  // Flag to track the position of "config.txt" argument

    while ((ch = getopt(argc, argv, "o:p:")) != -1)
    {
        switch (ch)
        {
        case 'o':
            strcpy(output_file, optarg);
            setenv("OUTPUT_FILE", output_file, 1);
            break;
        case 'p':
            strcpy(so_path, optarg);
            break;
        default:
            printf("Usage: ./logger config.txt [-o file] [-p sopath] command [arg1 arg2 ...]\n");
            return -1;
        }
    }

    char* argv_child[argc - optind + 1];  // Adjusted size for child process arguments
    int j = 0;

    for (int i = optind; i < argc; i++)
    {
        // Skip adding "config.txt" to argv_child
        if (strcmp(argv[i], "config.txt") == 0)
        {
            config_arg = 1;  // Set the flag to skip this argument
            continue;
        }

        argv_child[j] = malloc(strlen(argv[i]) + 1);  // +1 for null terminator
        strcpy(argv_child[j++], argv[i]);
    }

    // Add NULL terminator to the end of argv_child
    argv_child[j] = NULL;

    // Adjust the argument count if "config.txt" was skipped
    if (config_arg)
        j--;

    //fprintf(stderr, "so_path: %s\n", so_path);
    setenv("LD_PRELOAD", so_path, 1);
    execvp(argv_child[0], argv_child);
}

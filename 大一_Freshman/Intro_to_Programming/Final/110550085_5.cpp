#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _list {
	char c;
	int num;
}list;

int main(void) {
	char** ch = (char**)calloc(10000, sizeof(char*));
	char tiles[7];
	printf("tiles: ");
	scanf("%s", tiles);
	printf("number of possible non-empty sequences: ");
	if (strlen(tiles) == 1) {
		printf("1");
	}
	else if (strlen(tiles) == 2) {
		if (tiles[0] == tiles[1]) {
			printf("2");
		}
		else {
			printf("3");
		}
	}
	else if (strlen(tiles) == 3) {
		if (tiles[0] == tiles[1] && tiles[0] == tiles[2]) {
			printf("3");
		}
		else if ((tiles[0] == tiles[1] && tiles[0] != tiles[2]) || (tiles[0] != tiles[1] && tiles[0] == tiles[2]) || (tiles[1] == tiles[2] && tiles[0] != tiles[1])) {
			printf("8");
		}
		else {
			printf("15");
		}
	}
}
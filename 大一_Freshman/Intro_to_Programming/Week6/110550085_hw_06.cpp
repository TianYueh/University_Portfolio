#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
	FILE* finp;
	FILE* fout;
	finp=fopen("input.txt", "r");
	fout = fopen("output.txt", "w");
	int n;
	scanf("%d", &n);
	char arr[400][60];
	char s[100];
	for (int i = 0;i < n;i++) {
		scanf("%s", s);
		for (int j = 0;j < 60;j++) {
			arr[i][j] = s[j];
		}
	}
	char input[10000];
	while (fgets(input, 1005, finp)) {
		char tmp[400][60];
		for (int i = 0; i < 400; i++) {
			for (int j = 0; j < 60; j++) {
				tmp[i][j] = '\0';
			}
		}
		input[strlen(input)] = '\0';
		int last = 0;
		int m = 0;
		for (int i = 0;input[i];i++) {
			if (input[i] != ' ') {
				tmp[last][m] = input[i];
				m++;
			}
			else {
				tmp[last][m] = input[i];
				last++;
				m = 0;
			}
		}
		int kesu = 0;
		int no = 0;
		for (int i = 0;i <=last;i++) {
			for (int j = 0;j < n;j++) {
				for (int k = 0;k < strlen(arr[j]);k++) {
					if (arr[j][k] != tmp[i][k]) {
						no++;
					}
				}
				if (no==0) {
					fprintf(fout, "***");
					kesu++;
					if (tmp[i][(int)strlen(tmp[i]) - 2] == ',') {
						fprintf(fout, ",");
					}
					fprintf(fout, " ");
				}
				no = 0;
			}
			if (kesu == 0) {
				for (int p = 0;p < strlen(tmp[i]);p++) {
					fprintf(fout, "%c", tmp[i][p]);
				}
			}
			kesu = 0;
		}
	}
	
}
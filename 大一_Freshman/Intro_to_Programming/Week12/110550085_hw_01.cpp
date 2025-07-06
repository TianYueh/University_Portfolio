#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int main() {
	int m, n;
	printf("set the size of array A and B:");
	scanf("%d%d", &m, &n);
	char** A = (char**)calloc(m, sizeof(char*));
	char* a = (char*)calloc(100000,sizeof(char));
	for (int i = 0; i < m; i++) {
		*(A + i) = &a[100 * i];
	}
	char** B = (char**)calloc(n, sizeof(char*));
	char* b = (char*)calloc(100000,sizeof(char));
	for (int i = 0; i < n; i++) {
		*(B + i) = &b[100 * i];
	}
	
	while (1) {
		printf("0: add a book, 1:delete a book, 2: exchange the books\n");
		int num,toku=0;
		scanf("%d", &num);
		if (num == 0) {
			int x, y, s;
			scanf("%d%d%d", &x, &y, &s);
			if (x == 0) {
				if (y > m) {
					toku = 1;
				}
				if (y <= m && A[y][0] == NULL) {
					scanf("\n");
					for (int i = 0; i < s; i++) {
						char c;
						scanf("%c", &c);
						A[y][i] = c;
					}
				}
			}
			else if (x == 1) {
				if (y > n) {
					toku = 1;
				}
				if (y <= n && B[y][0] == NULL) {
					scanf("\n");
					for (int i = 0; i < s; i++) {
						char c;
						scanf("%c", &c);
						B[y][i] = c;
					}
				}
			}
		}
		else if (num == 1) {
			int x, y;
			scanf("%d%d", &x, &y);
			if (x == 0) {
				if (y > m) {
					toku = 1;
				}
				if (y <= m && A[y][0] != NULL) {
					for (int i = 0; i < strlen(A[y]); i++) {
						A[y][i] = '\0';
					}
				}
			}
			else if (x == 1) {
				if (y > n) {
					toku = 1;
				}
				if (y <= n && B[y][0] != NULL) {
					for (int i = 0; i < strlen(B[y]); i++) {
						B[y][i] = '\0';
					}
				}
			}
		}
		else if (num == 2) {
			int x, y;
			scanf("%d%d", &x, &y);
			if (x > m || y > n) {
				toku = 1;
			}
			if (x <=m && y <= n) {
				char* temp = (char*)calloc(100, sizeof(char));
				for (int i = 0; i < strlen(A[x]); i++) {
					temp[i] = A[x][i];
				}
				for (int i = 0; i < strlen(B[y]); i++) {
					A[x][i] = B[y][i];
				}
				for (int i = 0; i < strlen(A[x]); i++) {
					B[y][i] = temp[i];
				}
			}
		}
		if (toku == 0) {
			printf("A:\n");
			for (int i = 0; i < m; i++) {
				if (A[i][0] != NULL) {
					printf("%s\n", A[i]);
				}
				else {
					printf("(null)\n");
				}
			}
			printf("B:\n");
			for (int i = 0; i < n; i++) {
				if (B[i][0] != NULL) {
					printf("%s\n", B[i]);
				}
				else {
					printf("(null)\n");
				}
			}
		}
		else {
			toku = 0;
		}
	}
}
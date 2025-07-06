#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
	int m, n, p;
	printf("Please input M:");
	scanf("%d", &m);
	printf("Please input N:");
	scanf("%d", &n);
	printf("Please input P:");
	int dir = 0;
	scanf("%d", &p);
	int arr[32][32];
	int u = 0, l = 0;
	int r = n,d=m;
	int a = 0, b = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			arr[i][j] = 0;
		}
	}
	int ax = 0, ay = 0;
	for (int i = 1; i <= m * n; i++) {
		if (p == i) {
			ax = a;
			ay = b;
		}
		if (dir == 0) {
			arr[a][b] = i;
			b++;
			if (b == r-1) {
				r--;
				dir++;
			}
		}
		else if (dir == 1) {
			arr[a][b] = i;
			a++;
			if (a == d-1) {
				d--;
				dir++;
			}
		}
		else if (dir == 2) {
			arr[a][b] = i;
			b--;
			if (b == l) {
				l++;
				dir++;
			}
		}
		else if (dir == 3) {
			arr[a][b] = i;
			a--;
			if (a == u + 1) {
				u++;
				dir = 0;
			}
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%4d", arr[i][j]);
		}
		printf("\n");
	}
	if (p > m * n) {
		printf("P is out of range\n");
		return 0;
	}
	printf("The location of %d is:(%d,%d)",p,ax + 1, ay + 1);
}
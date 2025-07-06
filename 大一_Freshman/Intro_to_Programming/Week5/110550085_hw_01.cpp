#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
	int m, n, p;
	scanf("%d%d%d", &m, &n, &p);
	int amat[10][10] = { 0 }, bmat[10][10] = { 0 }, ans[10][10] = { 0 };
	for (int i = 0;i < m;i++) {
		for (int j = 0;j < n;j++) {
			scanf("%d", &amat[i][j]);
		}
	}
	printf("\n");
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < p;j++) {
			scanf("%d", &bmat[i][j]);
		}
	}
	for (int i = 0;i < m;i++) {
		for (int j = 0;j < p;j++) {
			for (int k = 0;k < n;k++) {
				ans[i][j] += (amat[i][k] * bmat[k][j]);
			}
		}
	}
	for (int i = 0;i < m;i++) {
		for (int j = 0;j < p;j++) {
			printf("%d ", ans[i][j]);
		}
		printf("\n");
	}
	system("pause");
	return 0;
}

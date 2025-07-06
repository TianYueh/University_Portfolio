#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

int main() {
	FILE* finp;
	FILE* fout;
	finp = fopen("input.txt", "r");
	fout = fopen("output.txt", "w");
	for (int i = 0; i < 5; i++) {
		unsigned long long int m,n,ans;
		fscanf(finp, "%d", &m);
		fscanf(finp, "%d", &n);
		ans = m * n;
		fprintf(fout, "%d\n", ans);
	}
}
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main() {
	printf("Input an integer:");
	int n,count=0,prime=0;
	scanf("%d", &n);
	printf("Output:\n");
	for (int i = 2; i <= n; i++) {
		prime = 0;
		for (int j = 2; j * j <= i; j++) {
			if (i%j == 0) {
				prime++;
			}
		}
		if (prime == 0) {
			printf("%d ", i);
			count++;
			if (count == 10) {
				count = 0;
				printf("\n");
			}
		}
	}
}
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
	FILE* finp;
	finp = fopen("q3.txt", "r");
	int n = 25000;
	int a[25000];
	for (int i = 0; i < n; i++) {
		fscanf(finp,"%d",&a[i]);
	}
	
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			if (a[i] < a[j]) {
				int temp = 0;
				temp = a[i];
				a[i] = a[j];
				a[j] = temp;
			}
		}
	}
	int sum = 0;
	for (int i = 0; i < n - 1; i++) {
		sum += a[i] - a[i + 1];
	}
	
	printf("The minimum value is equal to %d", sum);
}
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cstring>

int main(void) {
	int arr[7];
	printf("Original Unsorted array : ");
	for (int i = 0; i < 7; i++) {
		scanf("%d",&arr[i]);
	}
	int fb = 7;
	int f = 0;
	int b = 6;
	for (int k = 0; k < fb-1; k++) {
		if (k % 2 == 0) {
			printf("Forward Pass:\n");
			for (int i = f; i < b; i++) {
				if (arr[i] > arr[i + 1]) {
					int temp = 0;
					temp = arr[i];
					arr[i] = arr[i + 1];
					arr[i + 1] = temp;
					for (int j = 0; j < 7; j++) {
						printf("%d ", arr[j]);
					}
					printf("\n");
				}
			}
			b--;
		}
		else if (k % 2 == 1) {
			printf("Backward Pass:\n");
			for (int i = b; i > f; i--) {
				if (arr[i - 1] > arr[i]) {
					int temp = 0;
					temp = arr[i];
					arr[i] = arr[i - 1];
					arr[i - 1] = temp;
					for (int j = 0; j < 7; j++) {
						printf("%d ", arr[j]);
					}
					printf("\n");
				}
			}
			f++;
		}
	}
	printf("Sorted array : ");
	for (int i = 0; i < 7; i++) {
		printf("%d ", arr[i]);
	}
}
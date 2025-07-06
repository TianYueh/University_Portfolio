#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main() {
	FILE* finp;
	finp = fopen("p5.txt", "r");
	int n = 200;
	int arr[200];
	int ans = 0;
	for (int i = 0; i <n ; i++) {
		fscanf(finp, "%d",&arr[i]);
	}
	for(int i=0;i<n;i++){
		int min = 2147483647;
		int temp = 0;
		int place = 0;
		for (int j = i; j < n; j++) {
			if (arr[j] < min) {
				min = arr[j];
				place = j;
			}
		}
		temp = arr[place];
		arr[place] = arr[i];
		arr[i] = temp;
	}
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			for (int z = j + 1; z < n; z++) {
				if (arr[z] - arr[i] <= 2) {
					ans++;
				}
			}
		}
	}
	printf("The number of Close Tuples is %d", ans);
	
}

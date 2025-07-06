#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void bubbleSort(int a[], int size) {
	int temp, k, pass;
	for (int pass = 1; pass < size; pass++) {
		for (k = 0; k < size - pass; k++) {
			if (a[k] > a[k + 1]) {
				temp = a[k];
				a[k] = a[k + 1];
				a[k + 1] = temp;
			}
		}
	}
	return;
}

void selectionSort(int a[], int size) {
	int place=0,temp=0;
	for (int i = 0; i < size; i++) {
		int mini = 2147483647;
		for (int j = i; j < size; j++) {
			if (a[j] < mini) {
				mini = a[j];
				place = j;
			}
		}
		temp = a[i];
		a[i] = a[place];
		a[place] = temp;
	}
	return;
}

void insertionSort(int a[], int size) {
	int temp = 0;
	for (int i = 1; i < size; i++) {
		for (int j = 0; j <i; j++) {
			if (a[j] >= a[i]) {
				temp = a[i];
				for (int k = i; k >j; k--) {
					a[k] = a[k - 1];
				}
				a[j] = temp;
				break;
			}
		}
	}
	return;
}

int main(void) {
	FILE* finp;
	FILE* finpu;
	FILE* finput;
	int n = 25000;
	finp = fopen("data1.txt", "r");
	int* bubble = (int*)calloc(n, sizeof(int));
	for (int i=0; i < n; i++) {
		fscanf(finp,"%d", &bubble[i]);
	}
	finp = fopen("data1.txt", "r");
	int* selection = (int*)calloc(n, sizeof(int));
	for (int i = 0; i < n; i++) {
		fscanf(finp, "%d", &selection[i]);
	}
	finp = fopen("data1.txt", "r");
	int* insertion = (int*)calloc(n+1, sizeof(int));
	for (int i = 0; i < n; i++) {
		fscanf(finp, "%d", &insertion[i]);
	}
	double START, END,TIMEBUBBLE,TIMESELECTION,TIMEINSERTION;
	START = clock();
	bubbleSort(bubble, n);
	END = clock();
	TIMEBUBBLE = (END - START);
	printf("TIMEBUBBLE = %lf\n", TIMEBUBBLE / CLOCKS_PER_SEC);
	START = clock();
	selectionSort(selection, n);
	END = clock();
	TIMESELECTION = (END - START);
	printf("TIMESELECT = %lf\n", TIMESELECTION / CLOCKS_PER_SEC);
	START = clock();
	insertionSort(insertion, n);
	END = clock();
	TIMEINSERTION = (END - START);
	printf("TIMEINSERT = %lf\n", TIMEINSERTION / CLOCKS_PER_SEC);
	
	for (int i = 0; i < 25000; i++) {
		printf("%d ", bubble[i]);
		if (i % 10 == 9) {
			printf("\n");
		}
	}
	finpu = fopen("data2.txt", "r");
	for (int i = 0; i < n; i++) {
		fscanf(finpu, "%d", &bubble[i]);
	}
	finpu = fopen("data2.txt", "r");
	for (int i = 0; i < n; i++) {
		fscanf(finpu, "%d", &selection[i]);
	}
	finpu = fopen("data2.txt", "r");
	for (int i = 0; i < n; i++) {
		fscanf(finpu, "%d", &insertion[i]);
	}
	START = clock();
	bubbleSort(bubble, n);
	END = clock();
	TIMEBUBBLE = (END - START);
	printf("TIMEBUBBLE = %lf\n", TIMEBUBBLE / CLOCKS_PER_SEC);
	START = clock();
	selectionSort(selection, n);
	END = clock();
	TIMESELECTION = (END - START);
	printf("TIMESELECT = %lf\n", TIMESELECTION / CLOCKS_PER_SEC);
	START = clock();
	insertionSort(insertion, n);
	END = clock();
	TIMEINSERTION = (END - START);
	printf("TIMEINSERT = %lf\n", TIMEINSERTION / CLOCKS_PER_SEC);

	for (int i = 0; i < 25000; i++) {
		printf("%d ", selection[i]);
		if (i % 10 == 9) {
			printf("\n");
		}
	}
	finput = fopen("data3.txt", "r");
	
	for (int i = 0; i < n; i++) {
		fscanf(finput, "%d", &bubble[i]);
	}
	finput = fopen("data3.txt", "r");
	
	for (int i = 0; i < n; i++) {
		fscanf(finput, "%d", &selection[i]);
	}
	finput = fopen("data3.txt", "r");
	
	for (int i = 0; i < n; i++) {
		fscanf(finput, "%d", &insertion[i]);
	}
	START = clock();
	bubbleSort(bubble, n);
	END = clock();
	TIMEBUBBLE = (END - START);
	printf("TIMEBUBBLE = %lf\n", TIMEBUBBLE / CLOCKS_PER_SEC);
	START = clock();
	selectionSort(selection, n);
	END = clock();
	TIMESELECTION = (END - START);
	printf("TIMESELECT = %lf\n", TIMESELECTION / CLOCKS_PER_SEC);
	START = clock();
	insertionSort(insertion, n);
	END = clock();
	TIMEINSERTION = (END - START);
	printf("TIMEINSERT = %lf\n", TIMEINSERTION / CLOCKS_PER_SEC);
	for (int i = 0; i < 25000; i++) {
		printf("%d ", insertion[i]);
		if (i % 10 == 9) {
			printf("\n");
		}
	}
}
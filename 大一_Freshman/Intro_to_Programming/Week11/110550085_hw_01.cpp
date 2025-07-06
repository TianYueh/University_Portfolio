#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int* compare(const char* s, const char* t, int n) {
	int* cmp = (int*)calloc(n, sizeof(int));
	for (int i = 0; i < n; i++) {
		if (s[i] == t[i]) {
			cmp[i] = 1;
		}
		else {
			cmp[i] = 0;
		}
	}
	return cmp;
}

int main(void){
	int n;
	printf("input n:");
	scanf("%d", &n);
	char* s;
	char* t;
	s = (char*)calloc(n, sizeof(char));
	t = (char*)calloc(n, sizeof(char));
	printf("input a:");
	scanf("\n");
	for (int i = 0; i <= n; i++) {
		scanf("%c", &s[i]);
	}
	printf("input b:");
	for (int i = 0; i <= n; i++) {
		scanf("%c", &t[i]);
	}
	int* cmp = compare(s, t, n);
	for (int i = 0; i < n; i++) {
		printf("%d", cmp[i]);
	}
	return 0;
}
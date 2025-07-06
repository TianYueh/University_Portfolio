#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
	char s[16364] = "";
	scanf("%s", s);
	int length = strlen(s);
	int count = 0,cl=0;
	for (int i = 'A';i <= 'z';i++) {
		count = 0;
		cl = 0;
		for (int j = 0;j < length;j++) {
			if (s[j] == i) {
				if (count == 0) {
					printf("%c: ", i);
					count++;
					cl++;
				}
				printf("%d ", j);
			}
		}
		if (cl > 0) {
			printf("\n");
		}
	}
	system("pause");
	return 0;
}

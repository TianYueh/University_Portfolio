#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
	char s[40];
	int ans = 0;
	int aru = 0;
	int cul = 0;
	scanf("%s", s);
	for (int i = 0; i < strlen(s); i++) {
		if (s[i] == '(') {
			ans = 1;
			aru++;
			cul++;
		}
		else if (s[i] == ')') {
			if (cul == 0) {
				printf("False\n");
				return 0;
			}
			else {
				cul--;
			}
		}
	}
	if (cul == 0) {
		printf("True\n");
	}
	else {
		printf("False\n");
	}
}
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void transfer(void);
char ch;

void transfer() {
	if (ch >= 'A' && ch <= 'Z') {
		printf("%c", ch + 32);
	}
	else if (ch >= 'a' && ch <= 'z') {
		printf("%c", ch - 32);
	}
	else {
		printf("%c", ch);
	}

}

int main(void) {
	FILE* finp;
	finp = fopen("self_introduction.txt","r");
	char str[10000];
	if (finp != NULL) {
		while ((ch = getc(finp))!=EOF) {
			transfer();
		}
	}
	fclose(finp);
	return 0;
}
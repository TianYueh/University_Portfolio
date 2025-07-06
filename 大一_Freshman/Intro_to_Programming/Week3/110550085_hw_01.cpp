#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main(void) {
	while (1) {
		printf("ax^2+bx+c=0\n");
		printf("請輸入各項係數a、b、c\n");
		int a, b, c;
		scanf("%d %d %d", &a, &b, &c);
		printf("%dx^2+%dx+%d=0\n",a,b,c);
		if ((b * b - 4 * a * c) > 0) {
			printf("兩相異實數解\n\n");
		}
		else if ((b * b - 4 * a * c) == 0) {
			printf("兩相同實數解\n\n");
		}
		else if ((b * b - 4 * a * c) < 0) {
			printf("無實數解\n\n");
		}
	}
}

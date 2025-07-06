#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main(void) {
	printf("請輸入任一整數:");
	long long int n=0;
	scanf("%lld", &n);
	if (n < 0) {
		printf("小於此整數之所有完美數:\n");
		printf("無\n");
		system("pause");
		return 0;
	}
	else {
		printf("小於此整數之所有完美數:\n");
		for (int i = 1;i < n;i++) {
			int temp = 0;
			for (int j = 1;j < i;j++) {
				if (i % j == 0) {
					temp += j;
				}
			}
			if (temp == i) {
				printf("%d\n", i);
			}
		}
	}
	

}
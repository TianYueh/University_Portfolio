#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main() {
	int m, n;
	printf("輸入第一個正整數:");
	scanf("%d", &m);
	printf("輸入第二個正整數:");
	scanf("%d", &n);
	int ans = 0;
	for (int i = 1;i <= n;i++) {
		if (n % i == 0 && m % i == 0) {
			ans = i;
		}
	}
	printf("最大公因數:%d\n", ans);
	system("pause");
	return 0;
}

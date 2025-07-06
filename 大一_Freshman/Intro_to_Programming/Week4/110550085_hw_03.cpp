#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
	srand(time(NULL));
	int n = rand() % 101;
	printf("Key:%d\n", n);
	int left = 1, right = 100,times=1,guess=0;
	while (1) {
		printf("(第%d次猜測) %d ~ %d：",times,left,right);
		scanf("%d", &guess);
		if (guess == n) {
			printf("Bingo!\nKey:%d\n",n);
			break;
		}
		else if (guess > right||guess<left) {
			printf("超出範圍\n");
		}
		else if (guess < n) {
			printf("太小\n");
			left = guess;
		}
		else {
			printf("太大\n");
			right = guess;
		}
		times++;
	}
	system("pause");
	return 0;
}
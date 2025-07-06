#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <math.h>

void hanoi(int n, char a, char b, char c) {
	if (n == 1) {
		printf("Move disk %d from %c to %c\n", n,a,c);
	}
	else {
		hanoi(n - 1, a, c, b);
		printf("Move disk %d from %c to %c\n", n,a,c);
		hanoi(n - 1, b, a, c);
	}
}

int main(void) {
	int n;
	while (1){
		char A = 'A', B = 'B', C = 'C';
		printf("Input the number of disks: ");
		scanf("%d", &n);
		if (n == 0) {
			printf("No movement.\n\n");
		}
		else {
			hanoi(n, A, B, C);
			printf("\n");
		}
	}
	

}
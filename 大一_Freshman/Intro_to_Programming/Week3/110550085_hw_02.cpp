#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main(void) 
{
	while (1) {
		printf("Please input month: ");
		int n;
		scanf("%d", &n);
		switch (n) {
			case 3:
			case 4 : 
			case 5 : 
				printf("Spring!\n");
				break;
			case 6:
			case 7 :
			case 8 :
				printf("Summer!\n");
				break;
			case 9:
			case 10 :
			case 11 :
				printf("Autumn!\n");
				break;
			case 12:
			case 1:
			case 2:
				printf("Winter!\n");
				break;
			default:
				printf("錯誤輸入\n");
				break;
		}
	}
}

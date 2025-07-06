#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int tempC2F(int);

int tempC2F(int c) {
	int f = 0;
	f = 9 * c / 5 + 32;
	return f;
}

int main(void) {
	int C=0;
	printf("What's the temperature <Celsius>: ");
	scanf("%d", &C);
	printf("Celsius = %d, Fahrenheit = %d\n",C,tempC2F(C));
}
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main() {
	int a, b;
	float c;
	printf("身高:");
	scanf("%d", &a);
	printf("體重:");
	scanf("%d", &b);
	printf("BMI:");
	float m;
	m = (float)(a * a);
	m /= 10000;
	c = (float)(b/m);
	printf("%.2f", c);
	printf("\n四捨五入後的BMI:");
	int d = (int)(c + 0.5);
	float e = (float)d;
	printf("%.2f", e);

	

}
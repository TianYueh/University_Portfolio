#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cmath>

int main() {
	double x1, y1, x2, y2, x3, y3;
	double a, b, c;
	printf("Input the first point in R^2 space x,y= ");
	scanf("%lf%lf", &x1, &y1);
	printf("Input the second point in R^2 space x,y= ");
	scanf("%lf%lf", &x2, &y2);
	printf("Input the third point in R^2 space x,y= ");
	scanf("%lf%lf", &x3, &y3);
	printf("Output:\n");
	a = (double)((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	b= (double)((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
	c= (double)((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3));
	int temp = 0;
	if (a < b) {
		temp = a;
		a=b;
		b = temp;
	}
	if (b < c) {
		temp = b;
		b=c;
		c = temp;
	}
	if (a < b) {
		temp = a;
		a = b;
		b = temp;
	}
	
	if ((x1 == x2 && y1 == y2) || (x2 == x3 && y2 == y3) || (x1 == x3 && y1 == y3)) {
		printf("These three point cannot form a triangle.\n");
		return 0;
	}
	else if ((y2 - y1) / (x2 - x1) == (y3 - y2) / (x3 - x2)) {
		printf("These three point cannot form a triangle.\n");
		return 0;
	}
	else if (sqrt(a) > sqrt(b) + sqrt(c)) {
		printf("These three point cannot form a triangle.\n");
		return 0;
	}
	else if ((x1 == x2) || (x2 == x3) || (x1 == x3)) {
		if (a - b - c < 0.0001 && a - b - c>0) {
			printf("These three point form a right triangle.\n");
			return 0;
		}
		else if (a - b - c > -0.0001 && a - b - c < 0) {
			printf("These three point form a right triangle.\n");
			return 0;
		}
		else if (a - b - c == 0) {
			printf("These three point form a right triangle.\n");
		}
		else if (a > b + c) {
			printf("These three point form an obtuse triangle.\n");
			return 0;
		}
		else if (a < b + c) {
			printf("These three point form an acute triangle.\n");
			return 0;
		}
	}
	else if (a - b - c == 0) {
		printf("These three point form a right triangle.\n");
	}
	else if (a - b - c < 0.0001 && a - b - c>0) {
		printf("These three point form a right triangle.\n");
		return 0;
	}
	else if (a - b - c > -0.0001 && a - b - c < 0) {
		printf("These three point form a right triangle.\n");
		return 0;
	}
	else if (a > b + c ) {
		printf("These three point form an obtuse triangle.\n");
	}
	else if (a< b + c) {
		printf("These three point form an acute triangle.\n");
	}
}
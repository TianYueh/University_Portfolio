#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int count = 0;
double ans = 0;

double add(double current,double add) {
	return current + add;
}

double sub(double current, double sub) {
	return current - sub;
}

double mul(double current, double mul){
	return current * mul;
}

double div(double current, double div) {
	return current / div;
}

int quit = 0;

char* c = (char*)calloc(10001, sizeof(char));

double callback_execution(double* params, double(**callbacks)(double, double), int num_of_callback) {
	int number = 0;
	for (int i = 0; i < num_of_callback; i++) {
		if (i == 0) {
			ans = params[i];
		}
		if (c[i] == '+') {
			number = 0;
		}
		else if (c[i] == '-') {
			number = 1;
		}
		else if (c[i] == '*') {
			number = 2;
		}
		else if (c[i] == '/') {
			number = 3;
		}
		else if (c[i] == 'q') {
			return ans;
		}
		ans = callbacks[number](ans, params[i + 1]);
	}
}

int main(void) {
	int num_of_callback = 0;
	int num;
	while (1) {
		printf("Enter whether to continue or quit(1,0): ");
		scanf("%d", &num);
		if (num == 0) {
			return 0;
		}
		else if(num==1) {
			double callback_execution(double*, double(**)(double, double), int);
			double* params = (double*)calloc(10000,sizeof(double));
			double(**callbacks)(double,double) = (double(**)(double,double))calloc(4, sizeof(double*));
			callbacks[0] = add;
			callbacks[1] = sub;
			callbacks[2] = mul;
			callbacks[3] = div;
			while(1) {
				printf(" %d Enter parameter and function code(+,-,*,/): ", count);
				scanf("%lf %c", &params[count],&c[count]);
				if (c[count]=='q') {
					break;
				}
				count++;
			}
			ans = callback_execution(params, callbacks, count);
			printf("Final Reuslt: %lf\n", ans);
			for (int i = 0; i < count; i++) {
				c[count] = '\0';
			}
			count = 0;
			ans = 0;
		}
		else {
			printf("Omae wa mou shinteiru.\n");
		}
	}
}

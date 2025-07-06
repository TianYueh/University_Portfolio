#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct Planet_ {
	char name[20];
	float attr[4];
}Planet;

int main(void) {
	FILE* finp;
	finp = fopen("q2.txt", "r");
	printf("please input the number of data.\n");
	int num = 0;
	Planet pl[8];
	scanf("%d", &num);
	for (int i = 0; i < num; i++) {
		fscanf(finp, "%s%f%f%f%f", &pl[i].name,&pl[i].attr[0],&pl[i].attr[1],&pl[i].attr[2],&pl[i].attr[3]);
	}
	printf("please input the chosen attribute.\n");
	int at = 0;
	scanf("%d", &at);
	printf("please input the value you want.\n");
	double val = 0;
	scanf("%lf", &val);
	int ans = 0;
	double now = 0;
	for (int i = 0; i < num; i++) {
		if (i == 0) {
			now = pl[i].attr[at] - val;
			if (now < 0);
			now = -now;
		}
		if (val > pl[i].attr[at]) {
			if (val - pl[i].attr[at] < now) {
				now = val - pl[i].attr[at];
				
				ans = i;
			}
		}
		else {
			if (pl[i].attr[at] - val < now) {
				now = pl[i].attr[at] - val;
				ans = i;
			}
		}
	}
	printf("%s %e %e %e %e", pl[ans].name, pl[ans].attr[0], pl[ans].attr[1], pl[ans].attr[2], pl[ans].attr[3]);
}
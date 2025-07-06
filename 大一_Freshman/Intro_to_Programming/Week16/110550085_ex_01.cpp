#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct _patient {
	char name[100];
	int sex;
	double inf[4];
}patient;

int main() {
	printf("please input the size of data.\n");
	int size;
	scanf("%d", &size);
	FILE* finp;
	finp = fopen("patients_data.txt", "r");
	patient* pat = (patient*)calloc(size, sizeof(patient));
	for (int i = 0; i < size; i++) {
		fscanf(finp, "%s%d%lf%lf%lf%lf", &pat[i].name, &pat[i].sex, &pat[i].inf[0], &pat[i].inf[1], &pat[i].inf[2], &pat[i].inf[3]);
	}
	int sexuality;
	printf("please input which sex we want to select.\n");
	scanf("%d", &sexuality);
	int attribute;
	printf("please input which attribute we want to choose.\n");
	scanf("%d", &attribute);
	printf("please input the range of concern.\n");
	double min, max;
	scanf("%lf%lf", &min, &max);
	for (int i = 0; i < size; i++) {
		if (sexuality == 0) {
			if (pat[i].sex == sexuality) {
				if (pat[i].inf[attribute] > min && pat[i].inf[attribute] < max) {
					printf("%s %d %lf %lf %lf %lf\n", pat[i].name, pat[i].sex, pat[i].inf[0], pat[i].inf[1], pat[i].inf[2], pat[i].inf[3]);
				}
			}
		}
		else if (sexuality == 1) {
			if (pat[i].sex == sexuality) {
				if (pat[i].inf[attribute] > min && pat[i].inf[attribute] < max) {
					printf("%s %d %lf %lf %lf %lf\n", pat[i].name, pat[i].sex, pat[i].inf[0], pat[i].inf[1], pat[i].inf[2], pat[i].inf[3]);
				}
			}
		}
		else if (sexuality == 2) {
			if (pat[i].inf[attribute] > min && pat[i].inf[attribute] < max) {
				printf("%s %d %lf %lf %lf %lf\n", pat[i].name, pat[i].sex, pat[i].inf[0], pat[i].inf[1], pat[i].inf[2], pat[i].inf[3]);
			}
		}
	}
}

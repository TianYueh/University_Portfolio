#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct _customer {
	int id;
	int money;
}customer;

int main(void) {
	int num_of_rows = 0;
	FILE* finp = fopen("input.txt", "r");
	fscanf(finp,"%d", &num_of_rows);
	char attribute[100];
	for (int i = 0; i < 4; i++) {
		fscanf(finp, "%s", attribute);
	}
	customer* cus = (customer*)calloc(num_of_rows, sizeof(customer));
	for (int i = 0; i < num_of_rows; i++) {
		int size = 0;
		fscanf(finp, "%d", &size);
		if (size == 1) {
			int n;
			fscanf(finp, "%d", &n);
			for (int j = n; j < num_of_rows-1; j++) {
				cus[j].id = cus[j + 1].id;
				cus[j].money = cus[j + 1].money;
			}
		}
		else if (size == 2) {
			for (int j = 0; j < num_of_rows; j++) {
				if (cus[j].id == NULL) {
					fscanf(finp, "%d", &cus[j].id);
					fscanf(finp, "%d", &cus[j].money);
					break;
				}
			}
		}
		else if (size == 3) {
			int n,id,money;
			fscanf(finp, "%d%d%d", &id,&money,&n);
			for (int j = num_of_rows - 1; j > n; j--) {
				cus[j].id = cus[j - 1].id;
				cus[j].money = cus[j - 1].money;
			}
			cus[n].id = id;
			cus[n].money = money;
		}
	}
	for (int i = 0; i < num_of_rows; i++) {
		if (cus[i].id != NULL) {
			printf("ID money: %d %d\n", cus[i].id, cus[i].money);
		}
		else{
			break;
		}
	}
}
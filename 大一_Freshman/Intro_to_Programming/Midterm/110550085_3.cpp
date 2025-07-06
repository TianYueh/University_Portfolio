#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cmath>

int main(void) {
	int n;
	printf("Roman numeral: ");
	char arr[20];
	int ans=0;
	scanf("%s", arr);
	printf("This roman numeral is ");
	for (int i = 0; i < strlen(arr); i++) {
		if (arr[i] == 'M') {
			ans += 1000;
		}
		else if (arr[i] == 'C') {
			if (arr[i + 1] == 'D') {
				ans += 400;
				i++;
			}
			else if (arr[i + 1] == 'M') {
				ans += 900;
				i++;
			}
			else {
				ans += 100;
			}
		}
		else if (arr[i] == 'D') {
			ans += 500;
		}
		else if (arr[i] == 'X') {
			if (arr[i + 1] == 'L') {
				ans += 40;
				i++;
			}
			else if (arr[i + 1] == 'C') {
				ans += 90;
				i++;
			}
			else {
				ans += 10;
			}
		}
		else if (arr[i] == 'L') {
			ans += 50;
		}
		else if (arr[i] == 'I') {
			if (arr[i + 1] == 'X') {
				ans += 9;
				i++;
			}
			else if (arr[i + 1] == 'V') {
				ans += 4;
				i++;
			}
			else {
				ans++;
			}
		}
		else if (arr[i] == 'V') {
			ans += 5;
		}
	}
	printf("%d.", ans);
}
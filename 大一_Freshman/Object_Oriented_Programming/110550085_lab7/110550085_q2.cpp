#include <iostream>
using namespace std;

/* Add whatever you want*/
int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int n;
    int time=0;
    cin>>n;
    int s[1005];
    for(int i=0;i<n;i++){
        cin>>s[i];
    }
    for(int i=0;i<n;i++){
        int mini=0;
        int k=0;
        for(int j=i;j<n;j++){
            if(j==i){
                mini=s[j];
                k=j;
            }
            if(s[j]<mini){
                mini=s[j];
                k=j;
            }
        }
        if(i!=k){
            int temp=0;
            temp=s[k];
            s[k]=s[i];
            s[i]=temp;
            time++;
        }
    }
    cout<<time;
    return 0;
}

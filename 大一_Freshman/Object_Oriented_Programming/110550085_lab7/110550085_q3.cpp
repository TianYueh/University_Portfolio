#include <iostream>
using namespace std;

/* Add whatever you want*/
int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int s[1005];
    int time=0;
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>s[i];
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(s[j+1]<s[j]){
                int temp;
                temp=s[j+1];
                s[j+1]=s[j];
                s[j]=temp;
                time++;
            }
        }
    }
    for(int i=0;i<n;i++){
        cout<<s[i]<<" ";
    }
    return 0;
}

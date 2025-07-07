#include <iostream>
#include <cstring>
using namespace std;

int main(){
    string s;
    int arr[26]={0};
    getline(cin, s);
    for(int i=0;i<s.length();i++){
        arr[s[i]-'A']++;
    }
    for(int i='A';i<='Z';i++){
        printf("%c: ", i);
        cout<<arr[i-'A']<<"\n";
    }

}
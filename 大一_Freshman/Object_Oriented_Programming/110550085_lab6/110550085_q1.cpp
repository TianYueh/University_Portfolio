#include <iostream>
#include <stack>
#include <vector>
using namespace std;

/* Add whatever you want*/
int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    stack<long long int> s;
    vector<long long int> ans;
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        int k;
        cin>>k;
        if(k==0){
            long long int num;
            cin>>num;
            s.push(num);
        }
        else if(k==1){
            if(s.empty()){
                cout<<"Not legal"<<endl;
            }
            else{
                s.pop();
            }
        }
        else if(k==2){
            if(s.empty()){
                cout<<"Not legal"<<endl;
            }
            else{
                cout<<s.top()<<endl;
            }
        }
        else if(k==3){
            cout<<s.size()<<endl;
        }
    }
    if(s.empty()){
        cout<<"No data";
    }
    else{
        int j=s.size();
        for(int i=0;i<j;i++){
            ans.push_back(s.top());
            s.pop();
        }
        for(int i=ans.size()-1;i>=0;i--){
            cout<<ans[i]<<" ";
        }
    }


    return 0;
}

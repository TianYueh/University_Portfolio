#include <cmath>
#include <iostream>

using namespace std;


int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int table[19]={0};
    int qtable[19]={0};
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        int num;
        cin>>num;
        int fed=num%19;
        if(table[fed]==0){
            table[fed]=num;
        }
        else{
            int jet=1;
            while(jet<19){
                int cur=(fed+jet)%19;
                if(table[cur]==0){
                    table[cur]=num;
                    break;
                }
                else{
                    jet++;
                }
            }
        }

        if(qtable[fed]==0){
            qtable[fed]=num;
        }
        else{
            int pro=1;

            while(pro<19){
                int newn=(fed+pro*pro)%19;
                if(qtable[newn]==0){
                    qtable[newn]=num;
                    break;
                }
                else{
                    pro++;
                }
            }
        }
    }
    for(int i=0;i<19;i++){
        cout<<table[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<19;i++){
        cout<<qtable[i]<<" ";
    }
    return 0;
}

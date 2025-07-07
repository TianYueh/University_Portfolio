#include <iostream>
#include <vector>
using namespace std;

class customer{
    public:
        int team;
        int num;
};
/* Add whatever you want*/
int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int time=0;
    int n;
    while(cin>>n){
        if(n==0){
            return 0;
        }
        time++;
        cout<<"Scenario #"<<time<<endl;
        vector<customer> team;
        vector<customer> que;
        for(int i=0;i<n;i++){
            int num_of_teammember;
            cin>>num_of_teammember;
            for(int j=0;j<num_of_teammember;j++){
                customer c;
                c.team=i;
                cin>>c.num;
                team.push_back(c);
            }
        }
        string s;
        while(cin>>s){
            if(s=="STOP"){
                break;
            }
            else if(s=="ENQUEUE"){
                int bango;
                cin>>bango;
                if(que.empty()){
                    for(int m=0;m<team.size();m++){
                        if(team[m].num==bango){
                            que.push_back(team[m]);
                            auto iter = team.erase(team.begin()+m);
                        }
                    }
                }
                else{
                    for(int l=0;l<team.size();l++){
                        if(team[l].num==bango){
                            for(int m=que.size()-1;m>=0;m--){

                                if(que[m].team==team[l].team){
                                    if(m==que.size()-1){
                                        que.push_back(team[l]);
                                        auto iter = team.erase(team.begin()+l);
                                    }
                                    else{
                                        que.insert(que.begin()+m+1,team[l]);
                                        auto iter = team.erase(team.begin()+l);
                                    }
                                    break;
                                }
                                else if(m==0){
                                    que.push_back(team[l]);
                                    auto iter=team.erase(team.begin()+l);
                                }
                            }
                        }
                    }
                }
            }
            else if(s=="DEQUEUE"){
                cout<<que[0].num<<endl;
                auto iter=que.erase(que.begin());
            }
        }
        cout<<endl;
    }
    return 0;
}

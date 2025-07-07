#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

class edge{
    public:
    int lpt;
    int rpt;
    int weight;
};

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int n;
    cin>>n;
    int m;
    cin>>m;
    int ansnum=0;

    //cin the edges
    edge* arr=(edge*)calloc(m, sizeof(edge));
    for(int i=0;i<m;i++){
        cin>>arr[i].lpt>>arr[i].rpt>>arr[i].weight;
    }

    vector<edge> v;

    //sorting the edges by weight
    for(int i=0;i<m;i++){
        for(int j=i;j<m;j++){
            if(arr[i].weight>arr[j].weight){
                edge temp;
                temp=arr[i];
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }
    }

    //declaring the 2D array sets
    int** sets=(int**)calloc(n+1, sizeof(int*));
    for(int i=0;i<n+1;i++){
        sets[i]=(int*)calloc(n, sizeof(int));
    }
    for(int i=1;i<=n;i++){
        for(int j=0;j<n;j++){
            sets[i][j]=0;
        }
    }

    //makesets
    for(int i=1;i<=n;i++){
        sets[i][0]=i;
    }


    for(int i=0;i<m;i++){
        int l,r;
        for(int j=1;j<=n;j++){
            for(int k=0;k<n;k++){
                if(sets[j][k]==arr[i].lpt){
                    l=j;
                }
                else if(sets[j][k]==arr[i].rpt){
                    r=j;
                }
            }
        }
        if(l!=r){
            //union
            int ll=0, rl=0;
            for(int x=0;x<n;x++){
                if(sets[l][x]!=0){
                    ll++;
                }
                if(sets[r][x]!=0){
                    rl++;
                }
            }
            if(ll>=rl){
                for(int y=ll;y<ll+rl;y++){
                    sets[l][y]=sets[r][y-ll];
                }
                for(int y=0;y<rl;y++){
                    sets[r][y]=0;
                }
            }
            else{
                for(int y=rl;y<ll+rl;y++){
                    sets[r][y]=sets[l][y-rl];
                }
                for(int y=0;y<ll;y++){
                    sets[l][y]=0;
                }
            }


            //ansputting
            v.push_back(arr[i]);
            ansnum++;
        }
    }

    //sorting the ans
    for(int i=0;i<v.size();i++){
        for(int j=i;j<v.size();j++){
            if(v[i].lpt>v[j].lpt){
                edge temp;
                temp=v[i];
                v[i]=v[j];
                v[j]=temp;
            }
            else if(v[i].rpt>v[j].rpt){
                edge temp;
                temp=v[i];
                v[i]=v[j];
                v[j]=temp;
            }
        }
    }

    for(int i=0;i<ansnum;i++){
        cout<<v[i].lpt<<" "<<v[i].rpt<<" "<<v[i].weight<<endl;
    }
    return 0;
}


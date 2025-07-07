#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;

int mat[1005][1005];

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int n;
    cin>>n;
    queue<int> q;
    bool visited[1005];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cin>>mat[i][j];
        }
    }
    for(int i=0;i<n;i++){
        visited[i]=false;
    }

    for(int i=0;i<n;i++){
        if(visited[i]==false){
            q.push(i);
            visited[i]=true;
            cout<<i<<" ";
            while(!q.empty()){
                int i=q.front();
                q.pop();

                for(int j=0;j<n;j++){
                    if(mat[i][j]==1&&visited[j]==false){
                        q.push(j);
                        visited[j]=true;
                        cout<<j<<" ";
                    }
                }
            }
            cout<<endl;
        }
    }

    /*for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(mat[i][j]==1){
                v[i].vec.push(&v[i]);
                if(v[i].vec.size()>1){
                    v[i].isHaji=false;
                }
            }
        }
    }
    while(visited.size()!=n){
      for(int i=0;i<n;i++){
        if(v[i].isHaji=true){
            vertex* vcur=nullptr;
            vcur=&v[i];
            while(vcur.vec.size()!=0){
                vcur->visited=true;
                printf()
                vcur->vec.front
            }
        }
      }
    }
    */



    return 0;
}

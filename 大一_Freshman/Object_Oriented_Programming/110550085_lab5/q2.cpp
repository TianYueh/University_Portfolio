#include <iostream>
using namespace std;

class Room{
public:
    Room *left_room;
    Room *right_room;
    int index;
  //add constructor or functions if you need

};

class LL{
public:
    LL(){
        first=nullptr;
    }
    void push_back(int &_index){
        Room* newroom=new Room;
        newroom->index=_index;
        if(first==nullptr){
            newroom->left_room=nullptr;
            newroom->right_room=nullptr;
            first=newroom;
            current=newroom;
        }
        else{
            newroom->left_room=current;
            newroom->right_room=nullptr;
            current->right_room=newroom;
            current=newroom;
        }
    }
    void direction(int num){
        current=first;
        cout<<current->index<<" ";
        for(int i=0;i<num;i++){
            char c;
            cin>>c;
            if(c=='r'){
                if(current->right_room!=nullptr){
                    current=current->right_room;
                    cout<<current->index<<" ";
                }
                else{
                    cout<<"-1 ";
                }
            }
            else if(c=='l'){
                if(current->left_room!=nullptr){
                    current=current->left_room;
                    cout<<current->index<<" ";
                }
                else{
                    cout<<"-1 ";
                }
            }
        }
    }

private:
    Room* first;
    Room* current;
};

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int num;
    LL list;
    cin>>num;
    for(int i=0;i<num;i++){
        int index;
        cin>>index;
        list.push_back(index);
    }
    int num2;
    cin>>num2;
    list.direction(num2);
    return 0;
}

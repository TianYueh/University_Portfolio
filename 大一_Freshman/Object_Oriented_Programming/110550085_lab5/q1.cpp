#include <iostream>

using namespace std;

class Node
{
public:
    int data;
    Node* next;
    Node* prev;
};

class LinkedList
{
public:
    LinkedList(){
        first=nullptr;
        last=nullptr;
    }
    void push_front(int &_data)
    {
        Node* newnode=new Node;
        newnode->next=first;
        newnode->prev=NULL;
        newnode->data=_data;
        if(first!=nullptr){
            first->prev=newnode;
        }
        first=newnode;
        if(last==NULL){
            last=first;
        }
    }

    void push_back(int &_data)
    {
        Node* newnode=new Node;
        newnode->next=NULL;
        newnode->prev=last;
        newnode->data=_data;
        if(last!=nullptr){
            last->next=newnode;
        }
        last=newnode;
        if(first==NULL){
            first=last;
        }
    }

    void pop_front()
    {
        if(last==NULL)return;
        Node* newnode=first->next;
        if(newnode!=NULL){
            newnode->prev=NULL;
        }
        delete first;
        first=newnode;
    }

    void pop_back()
    {
        if(last==NULL)return;
        Node* newnode=last->prev;
        if(newnode!=NULL){
            newnode->next=NULL;
        }
        delete last;
        last=newnode;
    }

    void printList()
    {
        Node* current=last;
        while(current!=NULL){
            cout<<current->data<<" ";
            current=current->prev;
        }
    }
private:
// Hint: maintain head and tail node pointing to begin and end of the list
    Node* first;
    Node* last;
};

int main()
{
/* Hint: Read input from STDIN and perform the corresponding operation.*/
    LinkedList list;
    int num,operation;
    cin>>num;
    for(int i=0;i<num;i++){
        cin>>operation;
        if(operation==0){
            int data;
            cin>>data;
            list.push_front(data);
        }
        else if(operation==1){
            int data;
            cin>>data;
            list.push_back(data);
        }
        else if(operation==2){
            list.pop_front();
        }
        else if(operation==3){
            list.pop_back();
        }

    }
  	list.printList();
    return 0;
}

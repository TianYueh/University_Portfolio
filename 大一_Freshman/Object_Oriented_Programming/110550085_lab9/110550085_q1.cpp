#include <iostream>
#include <vector>
using namespace std;

class Node{
    public:
    int index;
    Node* LeftNode=nullptr;
    Node* RightNode=nullptr;
    int height;
};

int getHeight(Node* node){
    if(node==nullptr){
        return 0;
    }
    else{
        return node->height;
    }
}

int getBalance(Node* node){
    if(node==nullptr){
        return 0;
    }
    else{
        return getHeight(node->LeftNode)-getHeight(node->RightNode);
    }
}



int max(int a, int b){
    if(a>b){
        return a;
    }
    else{
        return b;
    }
}

Node* rightRotate(Node* cn){
    Node* a=cn->LeftNode;
    Node* T=a->RightNode;

    a->RightNode=cn;
    cn->LeftNode=T;

    cn->height=max(getHeight(cn->LeftNode),getHeight(cn->RightNode))+1;
    a->height=max(getHeight(a->LeftNode),getHeight(a->RightNode))+1;

    return a;
}

Node* leftRotate(Node* cn){
    Node* a=cn->RightNode;
    Node* T=a->LeftNode;

    a->LeftNode=cn;
    cn->RightNode=T;

    cn->height=max(getHeight(cn->RightNode),getHeight(cn->LeftNode))+1;
    a->height=max(getHeight(a->RightNode),getHeight(a->LeftNode))+1;

    return a;
}

Node* newNode(int num)
{
    Node* node = new Node;
    node->index = num;
    node->height = 1;
    return(node);
}

Node* insert(Node* node,int num){
    if(node==nullptr){
        return(newNode(num));
    }
    else{
        if(num<node->index){
            node->LeftNode=insert(node->LeftNode,num);
        }
        else{
            node->RightNode=insert(node->RightNode,num);
        }

        node->height=1+max(getHeight(node->LeftNode),getHeight(node->RightNode));

        int balance=getBalance(node);
        if(balance>1&&num<node->LeftNode->index){
            return rightRotate(node);
        }
        else if(balance<-1&&num>node->RightNode->index){
            return leftRotate(node);
        }
        else if(balance>1&&num>node->LeftNode->index){
            node->LeftNode=leftRotate(node->LeftNode);
            return rightRotate(node);
        }
        else if(balance<-1&&num<node->RightNode->index){
            node->RightNode=rightRotate(node->RightNode);
            return leftRotate(node);
        }

        return node;
    }
}

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int n;
    cin>>n;
    vector<int> v;
    for(int i=0;i<n;i++){
        int num;
        cin>>num;
        v.push_back(num);
    }
    Node* RootNode=nullptr;
    //Construction of the BinaryTree;
    for(int i=0;i<n;i++){
        RootNode=insert(RootNode,v[i]);

    }
    cout<<RootNode->index<<endl;
    return 0;
}

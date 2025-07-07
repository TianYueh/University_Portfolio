#include <iostream>
#include <vector>
#include <stack>
using namespace std;

// Enter your code here
class Node{
public:
    Node* leftchild=nullptr;
    Node* rightchild=nullptr;
    int num;
};

vector<int> preorder(Node* root){
    vector<int> r;
    stack<Node*> s;

    s.push(root);
    while(s.size()!=0){
        Node* node=s.top();
        s.pop();
        r.push_back(node->num);
        if(node->rightchild!=nullptr){
            s.push(node->rightchild);
        }
        if(node->leftchild!=nullptr){
            s.push(node->leftchild);
        }
    }
    return r;
}

vector<int> inorder(Node* root){
    vector<int> r;
    stack<Node*> s;
    Node* cur=root;

    while(cur!=nullptr||s.size()!=0){
        if(cur!=nullptr){
            s.push(cur);
            cur=cur->leftchild;
        }
        else{
            Node* node=s.top();
            s.pop();
            r.push_back(node->num);
            cur=node->rightchild;
        }
    }
    return r;
}

vector<int> postorder(Node* root){
    vector<int> r;
    stack<Node*> s;
    s.push(root);

    while(s.size()!=0){
        Node* node=s.top();
        if(node->leftchild==nullptr&&node->rightchild==nullptr){
            s.pop();
            r.push_back(node->num);
        }
        if(node->rightchild!=nullptr){
            s.push(node->rightchild);
            node->rightchild=nullptr;
        }
        if(node->leftchild!=nullptr){
            s.push (node->leftchild);
            node->leftchild=nullptr;
        }
    }
    return r;
}


int main(){
    // Enter your code here. Read input from STDIN. Print output to STDOUT
    vector<int> vec;
    int bango;
    while(cin>>bango){
        vec.push_back(bango);
    }
    Node* root=nullptr;
    Node* current=nullptr;
    int m;
    while(m<vec.size()){
        if(root==nullptr){
            Node* ano=new Node;
            ano->num=vec[m];
            root=ano;
        }
        else{
            current=root;

            while(1){
                if(vec[m]<current->num){
                    if(current->leftchild==nullptr){
                        Node* left=new Node;
                        left->num=vec[m];
                        current->leftchild=left;
                        break;
                    }
                    else{
                        current=current->leftchild;
                    }
                }
                else if(vec[m]>current->num){
                    if(current->rightchild==nullptr){
                        Node* right=new Node;
                        right->num=vec[m];
                        current->rightchild=right;
                        break;
                    }
                    else{
                        current=current->rightchild;
                    }
                }
            }
        }
        m++;
    }
    cout << "PreOrder" << endl;
    vector<int> pre=preorder(root);
    for(int i=0;i<pre.size();i++){
        cout<<pre[i]<<endl;
    }
    cout << "InOrder" << endl;
    vector<int> in=inorder(root);
    for(int i=0;i<in.size();i++){
        cout<<in[i]<<endl;
    }
    cout << "PostOrder" << endl;
    vector<int> post=postorder(root);
    for(int i=0;i<post.size();i++){
        cout<<post[i]<<endl;
    }


    return 0;
}

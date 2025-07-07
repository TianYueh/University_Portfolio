#include <iostream>
#include <cstdlib>
#include <vector>
#include <iterator>
using namespace std;
class BHeap {
   private:
   vector <int> heap;
   int l(int parent);
   int r(int parent);
   int par(int child);
   void heapifyup(int index);
   void heapifydown(int index);
   public:
      BHeap() {}
      void Insert(int element);
      void DeleteMin();
      int ExtractMin();
      void showHeap();
      int Size();
};
int main() {
   BHeap h;
   while (1) {
    int num;
    cin>>num;
    if(num==0){
        exit(0);
    }
    int value;
    int sum=0;
    for(int i=0;i<num;i++){
        cin>>value;
        h.Insert(value);
    }
    for(int i=0;i<num-1;i++){
        int e_one=h.ExtractMin();
        h.DeleteMin();
        int e_two=h.ExtractMin();
        h.DeleteMin();
        h.Insert(e_one+e_two);
        sum+=(e_one+e_two);
    }
    cout<<sum<<endl;

   }
   return 0;
}
int BHeap::Size() {
   return heap.size();
}
void BHeap::Insert(int ele) {
   heap.push_back(ele);
   heapifyup(heap.size() -1);
}
void BHeap::DeleteMin() {
   if (heap.size() == 0) {
      return;
   }
   heap[0] = heap.at(heap.size() - 1);
   heap.pop_back();
   heapifydown(0);
}
int BHeap::ExtractMin() {
   if (heap.size() == 0) {
      return -1;
   }
   else
   return heap.front();
}
void BHeap::showHeap() {
   vector <int>::iterator pos = heap.begin();
   while (pos != heap.end()) {
      cout<<*pos<<" ";
      pos++;
   }
   cout<<endl;
}
int BHeap::l(int parent) {
   int l = 2 * parent + 1;
   if (l < heap.size())
      return l;
   else
      return -1;
}
int BHeap::r(int parent) {
   int r = 2 * parent + 2;
   if (r < heap.size())
      return r;
   else
      return -1;
}
int BHeap::par(int child) {
   int p = (child - 1)/2;
   if (child == 0)
      return -1;
   else
      return p;
}
void BHeap::heapifyup(int in) {
   if (in >= 0 && par(in) >= 0 && heap[par(in)] > heap[in]) {
      int temp = heap[in];
      heap[in] = heap[par(in)];
      heap[par(in)] = temp;
      heapifyup(par(in));
   }
}
void BHeap::heapifydown(int in) {
   int child = l(in);
   int child1 = r(in);
   if (child >= 0 && child1 >= 0 && heap[child] > heap[child1]) {
      child = child1;
   }
   if (child > 0 && heap[in] > heap[child]) {
      int t = heap[in];
      heap[in] = heap[child];
      heap[child] = t;
      heapifydown(child);
   }
}

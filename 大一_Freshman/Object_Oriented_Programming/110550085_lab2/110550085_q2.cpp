#include <iostream>
using namespace std;

class calculator
{
	public:
      // TODO
      calculator(int &_a,int &_b,int &_c);
      calculator(int &_d,int &_e);
      calculator(int &_f);
      int add(){
          return *q+*r+*s;
      }
    private:
      int *q;
      int *r;
      int *s;
      int zero=0;
};

int main()
{
    int a, b, c, d, e, f;
  	calculator c1(a);
    calculator c2(b, c);
    calculator c3(d, e, f);
  	cin >> a >> b >> c >> d >> e >> f ;
    cout << c1.add() * c2.add() - c3.add() << endl;
    return 0;
}

calculator::calculator(int &_a,int &_b,int &_c){
    q=&_a;
    r=&_b;
    s=&_c;
}

calculator::calculator(int &_d,int &_e){
    q=&_d;
    r=&_e;
    s=&zero;
}

calculator::calculator(int &_f){
    q=&_f;
    r=&zero;
    s=&zero;
}

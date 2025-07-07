#include <iostream>
using namespace std;

class Calculator
{
    //TODO
    public:
        Calculator(int &_num1,int &_num2);
        void get_add_result(int &val);
        void get_sub_result(int &val);
        void get_mul_result(int &val);
        void get_div_result(int &val);
    private:
        int *__num1;
        int *__num2;
};

int main() {
    int num1 = 0;
    int num2 = 0;
    int value = 0;

    Calculator calc(num1, num2);
    while(cin >> num1 >> num2)
    {
        calc.get_add_result(value);
        cout << value << " ";
        calc.get_sub_result(value);
        cout << value << " ";
        calc.get_mul_result(value);
        cout << value << " ";
        calc.get_div_result(value);
        cout << value << endl;
    }

    return 0;
}

Calculator::Calculator(int &_num1,int &_num2)
{
    __num1=&_num1;
    __num2=&_num2;
}

void Calculator::get_add_result(int &val){
    val=*__num1+(*__num2);
}
void Calculator::get_sub_result(int &val){
    val=*__num1-(*__num2);
}
void Calculator::get_mul_result(int &val){
    val=*__num1*(*__num2);
}
void Calculator::get_div_result(int &val){
    val=*__num1/(*__num2);
}

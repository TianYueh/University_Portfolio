#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

class Student {
public:
	Student( ) { }
	string generate_address(){
    	//TODO
        cin>>first_name>>last_name>>dormitory>>ID;
        string s;
        s.append("1001 University Road, Hsinchu, Taiwan 300, ROC, dorm ");
        s.append(dormitory);
        s.append(", (");
        s.append(ID);
        s.append(") ");
        s.append(first_name);
        s.append(" ");
        s.append(last_name);
        return s;
    }

  	//Please implement the remain class
private:
	string first_name;
    string last_name;
	string dormitory;
	string ID;
};


int main() {
  	Student mStudent;
    /* Enter your code here. Read input from STDIN */

  	cout << mStudent.generate_address() << endl;
    return 0;
}



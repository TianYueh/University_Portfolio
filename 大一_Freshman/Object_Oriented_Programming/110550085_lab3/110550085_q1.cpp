#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
using namespace std;
//TODO:Finish the class, you can add thing you need
class Student {
public:
  	Student(string _name,string _text):__name(_name),__text(_text){
  	    WriteBlackboard(__name,__text);
  	};
  	static string SeeBlackBoard(){
        return _BlackBoard;
    };

private:
    string __name,__text;
    static string _BlackBoard;
  	static void WriteBlackboard(string __name,string __text){
        _BlackBoard=_BlackBoard+__name+":"+__text+"\n";
  	};
};

string Student::_BlackBoard="";

int main() {
  	string name,text;
  	vector<Student> Students;
  	while(cin>>name>>text){
    //TODO
        Student temp(name,text);
        Students.push_back(temp);
    }
  	cout <<Students.size()<<endl<<Student::SeeBlackBoard();
    return 0;
}



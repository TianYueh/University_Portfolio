#include "Object.h"

Object::Object(){
    //Intentionally void;
}



Object::Object(string na, string ta){
    setName(na);
    setTag(ta);
}

void Object::setName(string str){
    name=str;
}

void Object::setTag(string T){
    tag=T;
}

string Object::getName(){
    return name;
}

string Object::getTag(){
    return tag;
}

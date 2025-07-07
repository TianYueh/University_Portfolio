#include "Room.h"

Room::Room(bool b, int n, vector<Object*> obj){
    setIsExit(b);
    setIndex(n);
    setObjects(obj);
}

void Room::setUpRoom(Room* roomptr){
    upRoom=roomptr;
}

void Room::setDownRoom(Room* roomptr){
    downRoom=roomptr;
}

void Room::setLeftRoom(Room* roomptr){
    leftRoom=roomptr;
}

void Room::setRightRoom(Room* roomptr){
    rightRoom=roomptr;
}

void Room::setIsExit(bool b){
    isExit=b;
}

void Room::setIndex(int n){
    index=n;
}

void Room::setObjects(vector<Object*> obj){
    objects=obj;
}

vector<Object*> Room::getObjects(){
    return objects;
}

Room* Room::getUpRoom(){
    return upRoom;
}

Room* Room::getDownRoom(){
    return downRoom;
}

Room* Room::getLeftRoom(){
    return leftRoom;
}

Room* Room::getRightRoom(){
    return rightRoom;
}

bool Room::getIsExit(){
    return isExit;
}

int Room::getIndex(){
    return index;
}

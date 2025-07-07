#include "Item.h"

Item::Item(string s,string t,int hp,int atk,int def){
    setName(s);
    setTag(t);
    setHealth(hp);
    setAttack(atk);
    setDefense(def);
}

bool Item::triggerEvent(Object* ObjPtr){
    Player* playerPtr = static_cast<Player*>(ObjPtr);
    cout<<"\n*Hint: You got "<<this->getName()<<".*\n"<<endl;
    playerPtr->addItem(*this);
    return 0;
}

int Item::getHealth(){
    return health;
}

int Item::getAttack(){
    return attack;
}

int Item::getDefense(){
    return defense;
}

void Item::setAttack(int atk){
    attack=atk;
}

void Item::setDefense(int def){
    defense=def;
}

void Item::setHealth(int hp){
    health=hp;
}

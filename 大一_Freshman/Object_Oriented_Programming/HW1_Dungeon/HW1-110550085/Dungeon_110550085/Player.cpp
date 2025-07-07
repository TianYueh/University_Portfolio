#include "Player.h"

Player::Player(){
    //Intentionally void.
}


void Player::addItem(Item it){
    inventory.push_back(it);
}

void Player::increaseStates(int hp,int atk,int def){
    setMaxHealth(getMaxHealth()+hp);
    setAttack(getAttack()+atk);
    setDefense(getDefense()+def);
}

void Player::changeRoom(Room* roomptr){
    previousRoom=currentRoom;
    currentRoom=roomptr;
}

void Player::setCurrentRoom(Room* cur){
    currentRoom=cur;
}

void Player::setPreviousRoom(Room* pre){
    previousRoom=pre;
}

void Player::setInventory(vector<Item> vec){
    inventory=vec;
}

void Player::showInventory(){
    for(int i=0;i<inventory.size();i++){
        cout<<i+1<<". "<<inventory[i].getName()<<endl;
    }
}


void Player::setHelmet(Item* hel){
    cout<<endl<<"*The helmet is equipped.*"<<endl<<endl;
    Item it=*hel;
    Item* helptr=new Item(it);
    if(getHelmet()==nullptr){
        Helmet=helptr;
        increaseStates(helptr->getHealth(), helptr->getAttack(),helptr->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==helptr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                    inv[j]=inv[j+1];
                }
                inv.pop_back();
                setInventory(inv);
                break;
            }
        }
    }
    else{
        Item* ir=Helmet;
        Helmet=helptr;
        increaseStates(helptr->getHealth(), helptr->getAttack(),helptr->getDefense());
        increaseStates(-ir->getHealth(), -ir->getAttack(), -ir->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==helptr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                    inv[j]=inv[j+1];
                }
                inv.pop_back();
                inv.push_back(*ir);
                setInventory(inv);
                break;
            }
        }
    }

}

void Player::setChestplate(Item* che){
    cout<<endl<<"*The Chestplate is equipped.*"<<endl<<endl;
    Item it=*che;
    Item* cheptr=new Item(it);
    if(getChestplate()==nullptr){
        Chestplate=cheptr;
        increaseStates(cheptr->getHealth(), cheptr->getAttack(), cheptr->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==che->getName()){
                for(int j=i;j<inv.size()-1;j++){
                    inv[j]=inv[j+1];
                }
                inv.pop_back();
                setInventory(inv);
                break;
            }
        }
    }
    else{
        Item* ir=Chestplate;
        Chestplate=cheptr;
        increaseStates(cheptr->getHealth(), cheptr->getAttack(),cheptr->getDefense());
        increaseStates(-ir->getHealth(), -ir->getAttack(), -ir->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==cheptr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                inv[j]=inv[j+1];
            }
            inv.pop_back();
            inv.push_back(*ir);
            setInventory(inv);
            break;
            }
        }
    }

}

void Player::setLeftHand(Item* LH){
    cout<<endl<<"*"<<LH->getName()<<" is equipped.*"<<endl<<endl;
    Item it=*LH;
    Item* LHPtr=new Item(it);
    if(getLeftHand()==nullptr){
        LeftHand=LHPtr;
        increaseStates(LHPtr->getHealth(), LHPtr->getAttack(), LHPtr->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==LHPtr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                inv[j]=inv[j+1];
                }
                inv.pop_back();
                setInventory(inv);
                break;
            }
        }
    }
    else{
        Item* ir=LeftHand;
        LeftHand=LHPtr;
        increaseStates(LHPtr->getHealth(),LHPtr->getAttack(),LHPtr->getDefense());
        increaseStates(-ir->getHealth(), -ir->getAttack(), -ir->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==LHPtr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                inv[j]=inv[j+1];
            }
            inv.pop_back();
            inv.push_back(*ir);
            setInventory(inv);
            break;
            }
        }
    }

}

void Player::setRightHand(Item* RH){
    cout<<endl<<"*"<<RH->getName()<<" is equipped.*"<<endl<<endl;
    Item it=*RH;
    Item* RHPtr=new Item(it);
    if(getRightHand()==nullptr){
        RightHand=RHPtr;
        increaseStates(RHPtr->getHealth(), RHPtr->getAttack(),RHPtr->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==RHPtr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                    inv[j]=inv[j+1];
                }
                inv.pop_back();
                setInventory(inv);
                break;
            }
        }
    }
    else{
        Item* ir=RightHand;
        RightHand=RHPtr;
        increaseStates(RHPtr->getHealth(), RHPtr->getAttack(),RHPtr->getDefense());
        increaseStates(-ir->getHealth(), -ir->getAttack(), -ir->getDefense());
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            if(inv[i].getName()==RHPtr->getName()){
                for(int j=i;j<inv.size()-1;j++){
                    inv[j]=inv[j+1];
                }
                inv.pop_back();
                inv.push_back(*ir);
                setInventory(inv);
                break;
            }
        }
    }
}

Item* Player::getHelmet(){
    return Helmet;
}

Item* Player::getChestplate(){
    return Chestplate;
}

Item* Player::getLeftHand(){
    return LeftHand;
}

Item* Player::getRightHand(){
    return RightHand;
}

Room* Player::getCurrentRoom(){
    return currentRoom;
}

Room* Player::getPreviousRoom(){
    return previousRoom;
}

vector<Item> Player::getInventory(){
    return inventory;
}

void Player::openBackpack(){
    if(getInventory().size()==0){
        cout<<"\nThere's nothing in my backpack.\n"<<endl;
    }
    else{
        cout<<endl<<"What do I want to use?"<<endl;
        vector<Item> inv=getInventory();
        for(int i=0;i<inv.size();i++){
            cout<<i+1<<". "<<inv[i].getName()<<endl;
        }
        int chose;
        while(cin>>chose){
            if(chose>inv.size()){
                cout<<"Invalid choice."<<endl;
                break;
            }
            chose--;
            if(inv[chose].getTag()=="Helmet"){
                setHelmet(&inv[chose]);
                break;
            }
            else if(inv[chose].getTag()=="Chestplate"){
                setChestplate(&inv[chose]);
                break;
            }
            else if(inv[chose].getTag()=="LeftHand"){
                setLeftHand(&inv[chose]);
                break;
            }
            else if(inv[chose].getTag()=="RightHand"){
                setRightHand(&inv[chose]);
                break;
            }
            else{
                cout<<"Invalid choice."<<endl;
            }
        }
    }


}

bool Player::triggerEvent(Object* Objptr){
    Player* playerPtr=dynamic_cast<Player*>(Objptr);
    vector<Item> inv=getInventory();
        cout<<endl;
        cout<<"----------------------"<<endl;
        cout<<"I am "<<getName()<<"!"<<endl;
        cout<<"HP: "<<getCurrentHealth()<<"/"<<getMaxHealth()<<endl;
        cout<<"ATK: "<<getAttack()<<endl;
        cout<<"DEF: "<<getDefense()<<endl;
        cout<<endl;
        cout<<"Helmet:"<<endl;
        if(Helmet!=nullptr)
            cout<<getHelmet()->getName()<<endl;
        cout<<endl;
        cout<<"Chestplate:"<<endl;
        if(Chestplate!=nullptr)
            cout<<getChestplate()->getName()<<endl;
        cout<<endl;
        cout<<"LeftHand:"<<endl;
        if(LeftHand!=nullptr)
            cout<<getLeftHand()->getName()<<endl;
        cout<<endl;
        cout<<"RightHand:"<<endl;
        if(RightHand!=nullptr)
            cout<<getRightHand()->getName()<<endl;
        cout<<endl<<endl;



        cout<<"Items:"<<endl<<endl;
        for(int i=0;i<inv.size();i++){
            cout<<i+1<<". "<<inv[i].getName()<<endl;
        }
        cout<<"----------------------"<<endl;
        cout<<endl;
    return true;
}

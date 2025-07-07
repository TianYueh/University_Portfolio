#include "Monster.h"

Monster::Monster(string name,int hp,int atk,int def)
{
    setName(name);
    setTag("Monster");
    setCurrentHealth(hp);
    setAttack(atk);
    setDefense(def);
}



bool Monster::triggerEvent(Object* ObjPtr){
    int doublebreak = 0;
    Player* playerPtr = dynamic_cast<Player*>(ObjPtr);
    cout<<endl<<getName()<<" is ready to attack!"<<endl<<endl;
    while(getCurrentHealth()>0){
        if(playerPtr->checkIsDead()){
            cout<<"-----------------------"<<endl;
            cout<<"My eyes begin to be blurred."<<endl;
            cout<<"I begin to feel exhausted."<<endl;
            cout<<"I am falling down."<<endl;
            cout<<"The world seems to be great, huh?"<<endl;
            cout<<"But now it matters no more."<<endl<<endl;
            cout<<"You're dead, try harder!"<<endl;
            cout<<"-----------------------"<<endl;
            exit(0);
            return 1;
        }
        cout<<"What do I want to do?"<<endl;
        cout<<"1. Attack "<<getName()<<endl;
        cout<<"2. Retreat"<<endl;
        cout<<"3. Open my backpack"<<endl;
        int n;
        while(cin>>n){
            if(n==1){
                srand(time(NULL));
                int x=rand();
                int cri=0;
                x%=4;
                if(x==0){
                    cout<<endl<<"Critical Hit!"<<endl;
                    cri=1;
                }
                int damage=playerPtr->getAttack()-getDefense();
                if(damage<0){
                    damage=0;
                }
                if(cri==1){
                    damage*=2;
                }
                takeDamage(damage);
                cout<<endl<<getName()<<" has now "<<getCurrentHealth()<<" HP."<<endl;
                cout<<"My attack does "<<damage<<" damage."<<endl;
                int mondamage=getAttack()-playerPtr->getDefense();
                if(mondamage<0){
                    mondamage=0;
                }
                playerPtr->takeDamage(mondamage);
                cout<<"I have now "<<playerPtr->getCurrentHealth()<<" HP."<<endl<<endl;
                break;
            }
            else if(n==2){
                cout<<endl<<"Successively retreated!"<<endl<<endl;
                playerPtr->changeRoom(playerPtr->getPreviousRoom());
                doublebreak=1;
                break;
            }
            else if(n==3){
                vector<Item> inv=playerPtr->getInventory();
                if(inv.size()==0){
                    cout<<"\nThere's nothing in my backpack.\n"<<endl;
                    break;
                }
                cout<<"What do I want to use?"<<endl;
                playerPtr->showInventory();
                int number;
                cin>>number;
                number--;

                if(inv[number].getTag()=="HealingPotion"){
                    cout<<endl<<"I drink the healing potion."<<endl<<endl;
                    int ht=playerPtr->getCurrentHealth()+inv[number].getHealth();
                    if(ht>playerPtr->getMaxHealth()){
                        ht=playerPtr->getMaxHealth();
                    }
                    playerPtr->setCurrentHealth(ht);
                    for(int i=number;i<inv.size()-1;i++){
                        inv[i]=inv[i+1];
                    }
                    inv.pop_back();
                    playerPtr->setInventory(inv);
                }
                else if(inv[number].getTag()=="HurtingPotion"){
                    cout<<endl<<"I throw the hurting potion to the monster."<<endl<<endl;
                    takeDamage(inv[number].getHealth());
                    for(int i=number;i<inv.size()-1;i++){
                        inv[i]=inv[i+1];
                    }
                    inv.pop_back();
                    playerPtr->setInventory(inv);
                }
                else{
                    cout<<"I cannot use it now."<<endl;
                }
                break;
            }
        }
        if(doublebreak==1){
            break;
        }
    }
    if(doublebreak==0){
        cout<<getName()<<" is defeated!"<<endl<<endl;
        vector<Object*> vec;
        playerPtr->getCurrentRoom()->setObjects(vec);
    }

    return true;
}

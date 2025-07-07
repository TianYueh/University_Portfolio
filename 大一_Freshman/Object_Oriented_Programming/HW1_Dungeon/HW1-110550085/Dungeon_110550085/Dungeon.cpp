#include "Dungeon.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

using namespace std;

Dungeon::Dungeon(){
    //Intentionally void;
}

void Dungeon::runDungeon(){
    startGame();
    while(checkGameLogic()==0){
        vector<Object*> action=player.getCurrentRoom()->getObjects();
        chooseAction(action);
    }
}

void Dungeon::handleMovement(){
    cout<<"Where do I want to go?"<<endl;
    cout<<"1. Go up.";
    if(player.getCurrentRoom()->getUpRoom()==nullptr){
        cout<<"(Unable to go.)";
    }
    if(player.getCurrentRoom()->getIndex()==13){
        cout<<"(Requires a key.)";
    }
    cout<<endl;
    cout<<"2. Go down.";
    if(player.getCurrentRoom()->getDownRoom()==nullptr){
        cout<<"(Unable to go.)";
    }
    cout<<endl;
    cout<<"3. Go left.";
    if(player.getCurrentRoom()->getLeftRoom()==nullptr){
        cout<<"(Unable to go.)";
    }
    cout<<endl;
    cout<<"4. Go right.";
    if(player.getCurrentRoom()->getRightRoom()==nullptr){
        cout<<"(Unable to go.)";
    }
    cout<<endl;

    int where;
    while(cin>>where){
        if(where==1){
            if(player.getCurrentRoom()->getIndex()==13){
                int goable=0;
                vector<Item> ite=player.getInventory();
                for(int x=0;x<ite.size();x++){
                    if(ite[x].getTag()=="Key"){
                        goable=1;
                    }
                }
                if(goable==0){
                    cout<<"You don't have the Key."<<endl;
                    break;
                }
            }
            if(player.getCurrentRoom()->getUpRoom()==nullptr){
                cout<<"Unable to go."<<endl;
                continue;
            }

            else{
                player.setPreviousRoom(player.getCurrentRoom());
                player.setCurrentRoom(player.getCurrentRoom()->getUpRoom());
                break;
            }
        }
        else if(where==2){
            if(player.getCurrentRoom()->getDownRoom()==nullptr){
                cout<<"Unable to go."<<endl;
                continue;
            }

            else{
                player.setPreviousRoom(player.getCurrentRoom());
                player.setCurrentRoom(player.getCurrentRoom()->getDownRoom());
                break;
            }
        }
        else if(where==3){
            if(player.getCurrentRoom()->getLeftRoom()==nullptr){
                cout<<"Unable to go."<<endl;
                continue;
            }
            else{
                player.setPreviousRoom(player.getCurrentRoom());
                player.setCurrentRoom(player.getCurrentRoom()->getLeftRoom());
                break;
            }
        }
        else if(where==4){
            if(player.getCurrentRoom()->getRightRoom()==nullptr){
                cout<<"Unable to go."<<endl;
                continue;
            }
            else{
                player.setPreviousRoom(player.getCurrentRoom());
                player.setCurrentRoom(player.getCurrentRoom()->getRightRoom());
                break;
            }
        }
        else{
            cout<<"Invalid Input."<<endl;
        }
    }
}

void Dungeon::handleEvent(Object* objptr){
    if(objptr->getTag()=="NPC"){
        NPC* npcptr=static_cast<NPC*>(objptr);
        npcptr->triggerEvent(&player);
    }
    else if(objptr->getTag()=="Monster"){
        Monster* monptr=static_cast<Monster*>(objptr);
        monptr->triggerEvent(&player);
    }
}

void Dungeon::chooseAction(vector<Object*> action){
    int monava=0;
    cout<<"What do I want to do?"<<endl;
    for(int i=0;i<action.size()+3;i++){
        if(i==0){
            if(action.size()!=0){
                if(action[0]->getTag()=="Monster"){
                cout<<"1. Retreat."<<endl;
                monava=1;
                }
                else{
                    cout<<"1. Move."<<endl;
                }
            }
            else{
                cout<<"1. Move."<<endl;
            }
        }
        else if(i==1){
            cout<<"2. Check Status."<<endl;
        }
        else if(i==2){
            cout<<"3. Open my backpack."<<endl;
        }
        else if(i==3){
            if(action[0]->getTag()=="NPC"){
                NPC* iaction=dynamic_cast<NPC*>(action[0]);
                cout<<i+1<<". Talk to "<<iaction->getName()<<"."<<endl;
            }
            else if(action[0]->getTag()=="Monster"){
                Monster* iaction=dynamic_cast<Monster*>(action[0]);
                cout<<i+1<<". Fight with "<<iaction->getName()<<"."<<endl;
            }
            else if(action[0]->getTag()=="Key"){
                cout<<i+1<<". Open the Chest."<<endl;
            }
            else{
                i++;
            }
        }
    }
    int choseAction;
    while(cin>>choseAction){
        if(choseAction>action.size()+3||(choseAction<=0)){
            cout<<"Invalid Input."<<endl;
            continue;
        }
        else{
            if(choseAction==1){
                if(monava==1){
                    player.changeRoom(player.getPreviousRoom());
                    cout<<endl<<"Successfully retreated!"<<endl<<endl;
                }
                else{
                    handleMovement();
                }
                break;
            }
            else if(choseAction==2){
                player.triggerEvent(&player);
                break;
            }
            else if(choseAction==3){
                player.openBackpack();
                break;
            }
            else if(choseAction==4){
                if(action[0]->getTag()=="NPC"){
                    NPC* npcptr=dynamic_cast<NPC*>(action[0]);
                    npcptr->triggerEvent(&player);

                }
                else if(action[0]->getTag()=="Monster"){
                    Monster* monptr=dynamic_cast<Monster*>(action[0]);
                    monptr->triggerEvent(&player);
                }
                else if(action[0]->getTag()=="Key"){
                    Item* itemptr=dynamic_cast<Item*>(action[0]);
                    itemptr->triggerEvent(&player);
                    vector<Object*> blank;
                    player.getCurrentRoom()->setObjects(blank);
                }
                break;
            }
        }
    }
}

void Dungeon::createMap(){
    Item leatherhelmet("LeatherHelmet","Helmet",0,1,1);
    Item* LeatherHelmet=new Item(leatherhelmet);
    Item leatherchestplate("LeatherChestplate","Chestplate", 0, 1, 1);
    Item* LeatherChestplate=new Item(leatherchestplate);
    Item leatherleftglove("LeatherLeftGlove","LeftHand", 0, 1, 1);
    Item* LeatherLeftGlove=new Item(leatherleftglove);
    Item leatherrightglove("LeatherRightGlove","RightHand", 0, 1, 1);
    Item* LeatherRightGlove=new Item(leatherrightglove);
    Item naturehelmet("NatureHelmet","Helmet", 1, 2, 6);
    Item* NatureHelmet=new Item(naturehelmet);
    Item naturechestplate("NatureChestplate","Chestplate", 1, 2, 6);
    Item* NatureChestplate=new Item(naturechestplate);
    Item knife("Knife","RightHand", 0, 2, 0);
    Item* Knife=new Item(knife);
    Item ryuryu("RyuRyu","RightHand",5, 15, 1);
    Item* RyuRyu=new Item(ryuryu);
    Item tenkafubu("Tenkafubu","LeftHand",5, 15, 1);
    Item* Tenkafubu=new Item(tenkafubu);
    Item healingpotion_i("HealingPotion I","HealingPotion", 15, 0, 0);
    Item* HealingPotion_I=new Item(healingpotion_i);
    Item healingpotion_ii("HealingPotion II","HealingPotion", 30, 0, 0);
    Item* HealingPotion_II=new Item(healingpotion_ii);
    Item hurtingpotion_i("HurtingPotion I","HurtingPotion", 15, 0, 0);
    Item* HurtingPotion_I= new Item(hurtingpotion_i);
    Item hurtingpotion_ii("HurtingPotion II","HurtingPotion", 30, 0, 0);
    Item* HurtingPotion_II=new Item(hurtingpotion_ii);
    Item thekeytofinal("TheKeytoAWholeNewWorld","Key", 0, 0, 0);
    Item* TheKeytoFinal=new Item(thekeytofinal);
    Monster Slimy("Slimy", 10, 1, 0);
    Monster Gobly("Gobly", 20, 2, 2);
    Monster Zomby("Zomby", 25, 3, 1);
    Monster Flamia("Flamia", 30, 7, 3);
    Monster Blazia("Blazia", 40, 7, 4);
    Monster Rimuru("Rimuru", 50, 14, 4);
    Monster Mizuria("Mizuria", 30, 12, 3);
    Monster Guardian("Guardian", 70, 17, 5);
    Monster Boss("Boss-Flesrouy", 200, 20, 10);
    vector<Item> KiritoCom;
    KiritoCom.push_back(*LeatherHelmet);
    KiritoCom.push_back(*LeatherChestplate);
    KiritoCom.push_back(*LeatherLeftGlove);
    KiritoCom.push_back(*LeatherRightGlove);
    KiritoCom.push_back(*Knife);
    NPC Kirito("Kirito","\nI wake up in a dark room with only a lamp glowing dimly.\nAn old man sitting on a chair is looking at me with his eyes brimming in radiation.\n\"Uwo! Kimiwaatarashiiyuusyadesuka?\nMukashimukashi, oremokatsuteyuusyadeshitane.\"\nI shake my head to tell him that I don't even know a single word.\n\"Naruhotone, daga isekaidewanihongowomanabanakya. Korewo tabette.\"\nHe gave you something, it seems like that I should eat that to get to know what he's talking about.\nI eat the jelly, suddenly...\n\"Now you know what I am saying, huh?\"\nThe world is now in danger, you should go save the world.\n\"Keep going, and then you will see a tree.\"\n\"In this journey, you might encounter some monsters.\"\n\"Here, take some equipments.\"\n\"Don't be afraid, good luck!\"\n\n",KiritoCom);
    vector<Item> TreeCom;
    TreeCom.push_back(*NatureHelmet);
    TreeCom.push_back(*NatureChestplate);
    TreeCom.push_back(*HealingPotion_I);
    TreeCom.push_back(*HurtingPotion_I);
    NPC TreeSpirit("The Tree Spirit","\n\"Here you come, adventurer...\"\nThis sound comes from high above.\nI raise my head.\n\"I am the Tree Spirit.\"\n\"For years, I've been here to look for a man.\"\n\"Finally you are here to save us.\"\n\"Take my stuff and keep going!\"\n\"I recommend you to go left first.\"\n\n",TreeCom);
    vector<Item> WaterCom;
    WaterCom.push_back(*RyuRyu);
    WaterCom.push_back(*HealingPotion_II);
    WaterCom.push_back(*HurtingPotion_II);
    NPC WaterGod("Hideyoshi","\n\"Finally, here you come.\"\n\"Finally, I can see the end of this tragedy.\"\n\"Take this sword, keep going.\"\n\"I believe in you, adventurer!\"\n\n",WaterCom);
    vector<Item> FireCom;
    FireCom.push_back(*Tenkafubu);
    FireCom.push_back(*HealingPotion_II);
    FireCom.push_back(*HurtingPotion_II);
    NPC FireGod("Nobunaga","\n\"Finally, someone is here to succeed my dream.\"\n\"I am the God of fire, Nobunaga.\"\n\"For years, I've been waiting here.\"\n\"And today, destiny brings you here.\"\n\"Take them, good luck!\"\n\n",FireCom);

    vector<Object*> firstobj;
    NPC* KiritoPtr=new NPC(Kirito);
    firstobj.push_back(KiritoPtr);
    Room firstroom(false, 0, firstobj);
    Room* FirstRoom=new Room(firstroom);

    vector<Object*> secondobj;
    Monster* SlimyPtr=new Monster(Slimy);
    secondobj.push_back(SlimyPtr);
    Room secondroom(false, 1, secondobj);
    Room* SecondRoom=new Room(secondroom);

    vector<Object*> thirdobj;
    Monster* GoblyPtr=new Monster(Gobly);
    thirdobj.push_back(GoblyPtr);
    Room thirdroom(false, 2, thirdobj);
    Room* ThirdRoom=new Room(thirdroom);

    vector<Object*> fourthobj;
    Monster* ZombyPtr=new Monster(Zomby);
    fourthobj.push_back(ZombyPtr);
    Room fourthroom(false, 3, fourthobj);
    Room* FourthRoom=new Room(fourthroom);

    vector<Object*> treeobj;
    NPC* TreeSpiritPtr=new NPC(TreeSpirit);
    treeobj.push_back(TreeSpiritPtr);
    Room theroomwithatree(false, 4, treeobj);
    Room* TheRoomWithATree=new Room(theroomwithatree);

    vector<Object*> leftfirstobj;
    Monster* FlamiaPtr=new Monster(Flamia);
    leftfirstobj.push_back(FlamiaPtr);
    Room leftfirstroom(false, 5, leftfirstobj);
    Room* LeftFirstRoom=new Room(leftfirstroom);

    vector<Object*> leftsecondobj;
    Monster* BlaziaPtr=new Monster(Blazia);
    leftsecondobj.push_back(BlaziaPtr);
    Room leftsecondroom(false, 6, leftsecondobj);
    Room* LeftSecondRoom=new Room(leftsecondroom);

    vector<Object*> keyroomobj;
    keyroomobj.push_back(TheKeytoFinal);
    Room keyroom(false, 7, keyroomobj);
    Room* KeyRoom=new Room(keyroom);

    vector<Object*> firegodobj;
    NPC* FireGodPtr=new NPC(FireGod);
    firegodobj.push_back(FireGodPtr);
    Room firegodroom(false, 8, firegodobj);
    Room* FireGodRoom=new Room(firegodroom);

    vector<Object*> rightfirstobj;
    Monster* MizuriaPtr=new Monster(Mizuria);
    rightfirstobj.push_back(MizuriaPtr);
    Room rightfirstroom(false, 9, rightfirstobj);
    Room* RightFirstRoom=new Room(rightfirstroom);

    vector<Object*> rightsecondobj;
    Monster* RimuruPtr=new Monster(Rimuru);
    rightsecondobj.push_back(RimuruPtr);
    Room rightsecondroom(false, 10, rightsecondobj);
    Room* RightSecondRoom=new Room(rightsecondroom);

    vector<Object*> watergodobj;
    NPC* WaterGodPtr=new NPC(WaterGod);
    watergodobj.push_back(WaterGodPtr);
    Room watergodroom(false, 11, watergodobj);
    Room* WaterGodRoom=new Room(watergodroom);

    vector<Object*> guardianobj;
    Monster* GuardianPtr=new Monster(Guardian);
    guardianobj.push_back(GuardianPtr);
    Room guardianroom(false, 12, guardianobj);
    Room* GuardianRoom=new Room(guardianroom);

    vector<Object*> bossobj;
    Monster* BossPtr=new Monster(Boss);
    bossobj.push_back(BossPtr);
    Room bossroom(false, 13, bossobj);
    Room* BossRoom=new Room(bossroom);

    vector<Object*> winningobj;
    Room winningroom(true, 14, winningobj);
    Room* WinningRoom=new Room(winningroom);
    //Link start.
    FirstRoom->setUpRoom(SecondRoom);

    SecondRoom->setDownRoom(FirstRoom);
    SecondRoom->setUpRoom(ThirdRoom);

    ThirdRoom->setUpRoom(FourthRoom);
    ThirdRoom->setDownRoom(SecondRoom);

    FourthRoom->setUpRoom(TheRoomWithATree);
    FourthRoom->setDownRoom(ThirdRoom);

    TheRoomWithATree->setUpRoom(GuardianRoom);
    TheRoomWithATree->setDownRoom(FourthRoom);
    TheRoomWithATree->setLeftRoom(LeftFirstRoom);
    TheRoomWithATree->setRightRoom(RightFirstRoom);

    RightFirstRoom->setLeftRoom(TheRoomWithATree);
    RightFirstRoom->setRightRoom(RightSecondRoom);

    LeftFirstRoom->setLeftRoom(LeftSecondRoom);
    LeftFirstRoom->setRightRoom(TheRoomWithATree);

    RightSecondRoom->setLeftRoom(RightFirstRoom);
    RightSecondRoom->setRightRoom(WaterGodRoom);

    LeftSecondRoom->setLeftRoom(KeyRoom);
    LeftSecondRoom->setRightRoom(LeftFirstRoom);

    KeyRoom->setRightRoom(LeftSecondRoom);
    KeyRoom->setLeftRoom(FireGodRoom);

    WaterGodRoom->setLeftRoom(RightSecondRoom);

    FireGodRoom->setRightRoom(KeyRoom);

    GuardianRoom->setUpRoom(BossRoom);
    GuardianRoom->setDownRoom(TheRoomWithATree);
    BossRoom->setUpRoom(WinningRoom);
    BossRoom->setDownRoom(GuardianRoom);
    WinningRoom->setDownRoom(BossRoom);
    rooms.push_back(FirstRoom);
    rooms.push_back(SecondRoom);
    rooms.push_back(ThirdRoom);
    rooms.push_back(FourthRoom);
    rooms.push_back(TheRoomWithATree);
    rooms.push_back(LeftFirstRoom);
    rooms.push_back(LeftSecondRoom);
    rooms.push_back(KeyRoom);
    rooms.push_back(FireGodRoom);
    rooms.push_back(RightFirstRoom);
    rooms.push_back(RightSecondRoom);
    rooms.push_back(WaterGodRoom);
    rooms.push_back(GuardianRoom);
    rooms.push_back(BossRoom);
    rooms.push_back(WinningRoom);
}

void Dungeon::createPlayer(){
    string namae;
    cout<<"What's your name?(Type in)"<<endl;
    cin>>namae;
    player.setName(namae);
    player.setAttack(1);
    player.setDefense(1);
    player.setCurrentHealth(40);
    player.setMaxHealth(40);
    player.setCurrentRoom(rooms[0]);
    player.setPreviousRoom(nullptr);
    cout<<"Such a great name."<<endl<<endl;
    cout<<"------------------------"<<endl<<endl;
    cout<<"Girls are now praying."<<endl;
    cout<<"Your adventure is about to begin."<<endl<<endl;
    cout<<"------------------------"<<endl;
}

int Dungeon::checkGameLogic(){
    if(player.checkIsDead()){
        cout<<endl;
        cout<<"-------------------------"<<endl;
        cout<<"My eyes begin to be blurred."<<endl;
        cout<<"I begin to feel exhausted."<<endl;
        cout<<"I am falling down."<<endl;
        cout<<"The world seems to be great, huh?"<<endl;
        cout<<"But now it matters no more."<<endl<<endl;
        cout<<"You're dead, try harder!"<<endl;
        cout<<"-------------------------"<<endl;
        exit(0);
        return 1;
    }
    if(player.getCurrentRoom()->getIsExit()==true){
        cout<<endl;
        cout<<"-------------------------"<<endl;
        cout<<"I walk out to the whole new world."<<endl;
        cout<<"Birds are singing, flowers are blooming."<<endl;
        cout<<"My journey seems to be stopped for a while."<<endl;
        cout<<"But my story would never end."<<endl<<endl;
        cout<<"You won! Congratulations!"<<endl;
        cout<<"-------------------------"<<endl;
        exit(0);
        return 2;
    }
    else{
        return 0;
    }
}

void Dungeon::startGame(){
    createMap();
    createPlayer();
}

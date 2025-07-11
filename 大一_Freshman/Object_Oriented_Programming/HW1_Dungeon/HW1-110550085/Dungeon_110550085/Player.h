#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>
#include "GameCharacter.h"
#include "Room.h"
#include "Item.h"

using namespace std;

class Item;

class Dungeon;

class Player: public GameCharacter
{
private:
    Room* currentRoom;
    Room* previousRoom;
    vector<Item> inventory;
    Item* Helmet=nullptr;
    Item* Chestplate=nullptr;
    Item* LeftHand=nullptr;
    Item* RightHand=nullptr;
public:
    Player();
    //Player(string,int,int,int);
    void addItem(Item);
    void increaseStates(int,int,int);
    void changeRoom(Room*);
    void openBackpack();

    /* Virtual function that you need to complete   */
    /* In Player, this function should show the     */
    /* status of player.                            */
    bool triggerEvent(Object*);

    /* Set & Get function*/
    void setCurrentRoom(Room*);
    void setPreviousRoom(Room*);
    void setInventory(vector<Item>);
    void setHelmet(Item*);
    void setChestplate(Item*);
    void setLeftHand(Item*);
    void setRightHand(Item*);
    void showInventory();
    Item* getHelmet();
    Item* getChestplate();
    Item* getLeftHand();
    Item* getRightHand();
    Room* getCurrentRoom();
    Room* getPreviousRoom();
    vector<Item> getInventory();
};

#endif // PLAYER_H_INCLUDED

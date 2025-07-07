#include "GameCharacter.h"

GameCharacter::GameCharacter(){
    //Intensionally void;
}

GameCharacter::GameCharacter(string n, string t, int hp, int atk, int def){
    setName(n);
    setTag(t);
    setCurrentHealth(hp);
    setAttack(atk);
    setDefense(def);
}

bool GameCharacter::triggerEvent(Object* objptr){
    return false;
}

bool GameCharacter::checkIsDead(){
    if(currentHealth<=0){
        return true;
    }
    else{
        return false;
    }
}

void GameCharacter::takeDamage(int dmg){
    currentHealth-=dmg;
}

int GameCharacter::getAttack(){
    return attack;
}

int GameCharacter::getMaxHealth(){
    return maxHealth;
}

int GameCharacter::getCurrentHealth(){
    return currentHealth;
}

int GameCharacter::getDefense(){
    return defense;
}

void GameCharacter::setMaxHealth(int health){
    maxHealth=health;
}

void GameCharacter::setCurrentHealth(int health){
    currentHealth=health;
}

void GameCharacter::setAttack(int atk){
    attack=atk;
}

void GameCharacter::setDefense(int def){
    defense=def;
}


#include "NPC.h"

NPC::NPC(string n, string scr, vector<Item> it)
{
    setName(n);
    setTag("NPC");
    setScript(scr);
    setCommodity(it);
    setMaxHealth(100);
    setCurrentHealth(100);
    setAttack(0);
    setDefense(100);
}

void NPC::listCommodity(){
    vector<Item> com;
    com=getCommodity();

}

bool NPC::triggerEvent(Object* Objptr){
    Player* playerPtr=dynamic_cast<Player*>(Objptr);
    cout<<getScript();
    cout<<"\n\"You may take these.\""<<endl;
    listCommodity();
    vector<Item> com;
    com=getCommodity();

    int n;
    while(com.size()!=0){
        for(int i=0;i<com.size();i++){
            cout<<i+1<<". "<<com[i].getName()<<endl;
        }
        cin>>n;
        n--;
        if(n>=com.size()||n<0){
            continue;
        }
        com[n].triggerEvent(playerPtr);
        for(int k=n;k<com.size()-1;k++){
            com[k]=com[k+1];
        }
        com.pop_back();
    }
    vector<Object*> blank;
    blank.clear();
    playerPtr->getCurrentRoom()->setObjects(blank);
    return true;
}

void NPC::setScript(string scr){
    script=scr;
}

string NPC::getScript(){
    return script;
}

void NPC::setCommodity(vector<Item> Com){
    commodity=Com;
}

vector<Item> NPC::getCommodity(){
    return commodity;
}


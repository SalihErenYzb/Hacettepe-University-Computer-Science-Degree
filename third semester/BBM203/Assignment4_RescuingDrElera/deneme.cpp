#include <iostream>
#include "SpaceSectorBST.h"
#include <string>
using namespace std;
int main(){
    SpaceSectorBST bst;
    bst.readSectorsFromFile("sampleIO/sectors.dat");
    bst.displaySectorsInOrder();


}
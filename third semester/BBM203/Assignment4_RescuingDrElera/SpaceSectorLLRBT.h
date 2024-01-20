#ifndef SPACESECTORLLRBT_H
#define SPACESECTORLLRBT_H

#include "Sector.h"
#include <iostream>
#include <fstream>  
#include <sstream>
#include <vector>

class SpaceSectorLLRBT {
public:
    Sector* root;
    SpaceSectorLLRBT();
    ~SpaceSectorLLRBT();
    void readSectorsFromFile(const std::string& filename);
    void insertSectorByCoordinates(int x, int y, int z);
    void displaySectorsInOrderRecursively(Sector* root);
    void displaySectorsInOrder();
    void displaySectorsPreOrderRecursively(Sector* root);
    void displaySectorsPreOrder();
    void displaySectorsPostOrderRecursively(Sector* root);
    void displaySectorsPostOrder();
    void deleteBSTRecursively(Sector* root);
    void rotateLeft(Sector*&root);
    void rotateRight(Sector*& root);
    void flipColors(Sector* root);
    bool isRed(Sector* root);
    void helperGetStellarPath(Sector*& root, const std::string& sector_code, std::vector<Sector*>& mainpath);
    void insertSectorRecursively(Sector*& root, Sector* newNode);
    std::vector<Sector*> getStellarPath(const std::string& sector_code);
    void printStellarPath(const std::vector<Sector*>& path);
};

#endif // SPACESECTORLLRBT_H

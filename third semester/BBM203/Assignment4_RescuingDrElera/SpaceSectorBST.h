#ifndef SPACESECTORBST_H
#define SPACESECTORBST_H

#include <iostream>
#include <fstream>  
#include <sstream>
#include <vector>

#include "Sector.h"

class SpaceSectorBST {
  
public:
    Sector *root;
    SpaceSectorBST();
    ~SpaceSectorBST();
    void readSectorsFromFile(const std::string& filename); 
    void insertSectorByCoordinates(int x, int y, int z);
    void insertSectorRecursively(Sector*& root, Sector* newNode);
    void deleteSectorRecursively(Sector*& root, const std::string& sector_code);
    void deleteSector(const std::string& sector_code);
    void deleteBSTRecursively(Sector* root);
    void displaySectorsInOrderRecursively(Sector* root);
    void displaySectorsInOrder();
    void displaySectorsPreOrderRecursively(Sector* root);
    void displaySectorsPreOrder();
    void displaySectorsPostOrderRecursively(Sector* root);
    void displaySectorsPostOrder();
    void helperGetStellarPath(Sector*& root, const std::string& sector_code, std::vector<Sector*>& mainpath);

    std::vector<Sector*> getStellarPath(const std::string& sector_code);
    void printStellarPath(const std::vector<Sector*>& path);
};

#endif // SPACESECTORBST_H

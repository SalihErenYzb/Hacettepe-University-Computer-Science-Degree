#include "SpaceSectorLLRBT.h"
#include <algorithm>
using namespace std;

SpaceSectorLLRBT::SpaceSectorLLRBT() : root(nullptr) {}
void SpaceSectorLLRBT::deleteBSTRecursively(Sector* root){
    if (root == nullptr){
        return;
    }
    deleteBSTRecursively(root->left);
    deleteBSTRecursively(root->right);
    delete root;
}
bool SpaceSectorLLRBT::isRed(Sector* root){
    if (root == nullptr){
        return false;
    }
    return root->color == RED;
}
void SpaceSectorLLRBT::readSectorsFromFile(const std::string& filename) {
    // TODO: read the sectors from the input file and insert them into the LLRBT sector map
    ifstream file(filename);
    string line;
    getline(file, line);
    while (getline(file, line)){
        istringstream iss(line);
        std::string token;
        int x, y, z;
        std::getline(iss, token, ',');
        x = std::stoi(token);
        std::getline(iss, token, ',');
        y = std::stoi(token);
        std::getline(iss, token, ',');
        z = std::stoi(token);
        insertSectorByCoordinates(x, y, z);
    }
    file.close();
}

// Remember to handle memory deallocation properly in the destructor.
SpaceSectorLLRBT::~SpaceSectorLLRBT() {
    // TODO: Free any dynamically allocated memory in this class.
    deleteBSTRecursively(root);
}
void SpaceSectorLLRBT::rotateLeft(Sector*& root){
    Sector* temp = root->right;
    root->right = temp->left;
    if (temp->left != nullptr){
        temp->left->parent = root;
    }
    temp->left = root;
    temp->parent = root->parent;
    root->parent = temp;
    temp->color = root->color;
    root->color = RED;
    root = temp;
}
void SpaceSectorLLRBT::rotateRight(Sector*& root){
    Sector* temp = root->left;
    root->left = temp->right;
    if (temp->right != nullptr){
        temp->right->parent = root;
    }
    temp->right = root;
    temp->parent = root->parent;
    root->parent = temp;
    temp->color = root->color;
    root->color = RED;
    root = temp;
}
void SpaceSectorLLRBT::flipColors(Sector* root){
    root->color = !root->color;
    if (root->left != nullptr){
        root->left->color = !root->left->color;
    }
    if (root->right != nullptr){
        root->right->color = !root->right->color;
    }
}
void SpaceSectorLLRBT::insertSectorRecursively(Sector*& root, Sector* newNode) {
    if (root == nullptr) {
        root = newNode;
        return;
    }
    newNode->parent = root;
    if (*newNode < *root) {
        insertSectorRecursively(root->left, newNode);
    }
    else if (*root < *newNode ) {
        insertSectorRecursively(root->right, newNode);
    }else{  
        return;
    }
    if (isRed(root->right) && !isRed(root->left)) {
        rotateLeft(root);
    }
    if (isRed(root->left) && isRed(root->left->left)) {
        rotateRight(root);
    }
    if (isRed(root->left) && isRed(root->right)) {
        flipColors(root);
    }
}
void SpaceSectorLLRBT::insertSectorByCoordinates(int x, int y, int z) {
    // TODO: Instantiate and insert a new sector into the space sector LLRBT map 
    // according to the coordinates-based comparison criteria.
    Sector* newNode = new Sector(x, y, z);
    insertSectorRecursively(root, newNode);
    //find the root
    while (root->parent != nullptr){
        root = root->parent;
    }
    root->color = BLACK;
}

void SpaceSectorLLRBT::displaySectorsInOrderRecursively(Sector* root) {
    if (root == nullptr){
        return;
    }
    displaySectorsInOrderRecursively(root->left);
    if (root->color ==  RED){
        cout << "RED sector: ";
    }else{
        cout << "BLACK sector: ";
    }
    cout <<  root->sector_code << endl;

    displaySectorsInOrderRecursively(root->right);
}
void SpaceSectorLLRBT::displaySectorsInOrder() {
    // TODO: Traverse the space sector BST map in-order and print the sectors 
    // to STDOUT in the given format.
    cout << "Space sectors inorder traversal:" << endl;
    displaySectorsInOrderRecursively(root);
    cout << endl;
}
void SpaceSectorLLRBT::displaySectorsPreOrderRecursively(Sector* root) {
    if (root == nullptr){
        return;
    }
    if (root->color ==  RED){
        cout << "RED sector: ";
    }else{
        cout << "BLACK sector: ";
    }
    cout << root->sector_code << endl;

    displaySectorsPreOrderRecursively(root->left);
    displaySectorsPreOrderRecursively(root->right);
}
void SpaceSectorLLRBT::displaySectorsPreOrder() {
    // TODO: Traverse the space sector BST map in pre-order traversal and print 
    // the sectors to STDOUT in the given format.
    cout << "Space sectors preorder traversal:" << endl;
    displaySectorsPreOrderRecursively(root);
    cout << endl;
}
void SpaceSectorLLRBT::displaySectorsPostOrderRecursively(Sector* root) {
    if (root == nullptr){
        return;
    }
    displaySectorsPostOrderRecursively(root->left);
    displaySectorsPostOrderRecursively(root->right);
    if (root->color ==  RED){
        cout << "RED sector: ";
    }else{
        cout << "BLACK sector: ";
    }
    cout << root->sector_code << endl;
}
void SpaceSectorLLRBT::displaySectorsPostOrder() {
    // TODO: Traverse the space sector BST map in post-order traversal and print 
    // the sectors to STDOUT in the given format.
    cout << "Space sectors postorder traversal:" << endl;
    displaySectorsPostOrderRecursively(root);
    cout << endl;
}
void SpaceSectorLLRBT::helperGetStellarPath(Sector*& root, const std::string& sector_code,std::vector<Sector*>& mainpath){
    if (root == nullptr){
        return;
    }
    if (root->sector_code == sector_code){
        mainpath.push_back(root);
        Sector* temp = root;
        while (temp->parent != nullptr){
            mainpath.push_back(temp->parent);
            temp = temp->parent; 
        }
        return;
    }
    helperGetStellarPath(root->left, sector_code, mainpath);
    helperGetStellarPath(root->right, sector_code, mainpath);
}
std::vector<Sector*> SpaceSectorLLRBT::getStellarPath(const std::string& sector_code) {
    std::vector<Sector*> path;
    // TODO: Find the path from the Earth to the destination sector given by its
    // sector_code, and return a vector of pointers to the Sector nodes that are on
    // the path. Make sure that there are no duplicate Sector nodes in the path!
    helperGetStellarPath(root, sector_code, path);
    std::reverse(path.begin(), path.end());
    std::vector<Sector*> pathToEarth;
    helperGetStellarPath(root, "0SSS", pathToEarth);
    std::reverse(pathToEarth.begin(), pathToEarth.end());

    int i = 0;
    while (i < path.size() && i < pathToEarth.size() && path[i] == pathToEarth[i]){
        i++;
    }
    i--;
    std::vector<Sector*> finalPath;
    for (int j = i; j < pathToEarth.size(); j++){
        finalPath.push_back(pathToEarth[j]);
    }
    std::reverse(finalPath.begin(), finalPath.end());
    for (int j = i + 1; j < path.size(); j++){
        finalPath.push_back(path[j]);
    }
    return finalPath;   
}

void SpaceSectorLLRBT::printStellarPath(const std::vector<Sector*>& path) {
    // TODO: Print the stellar path obtained from the getStellarPath() function 
    // to STDOUT in the given format.
    if (path.size() == 0){
        cout << "A path to Dr. Elara could not be found." << endl;
        return;
    }
    cout << "The stellar path to Dr. Elara: ";
    for (int i = 0; i < path.size(); i++){
        cout << path[i]->sector_code;
        if (i != path.size()-1){
            cout << "->";
        }
    }
    cout << endl ;
}
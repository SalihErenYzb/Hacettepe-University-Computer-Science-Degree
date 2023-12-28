#include "Sector.h"
#include <cmath>
// Constructor implementation
std::string Sector::helperF(int num, std::string str1, std::string str2, std::string str3){
    if (num > 0){
        return str1;
    }else if (num == 0){
        return str2;
    }else{
        return str3;
    }
}
Sector::Sector(int x, int y, int z) : x(x), y(y), z(z), left(nullptr), right(nullptr), parent(nullptr), color(RED) {
        // TODO: Calculate the distance to the Earth, and generate the sector code
        // based on the coordinates and the distance.
        distance_from_earth = sqrt(pow(x,2)+pow(y,2)+pow(z,2));
        sector_code = std::to_string(static_cast<int>(std::floor(distance_from_earth)));
        sector_code += helperF(x, "R", "S", "L") + helperF(y, "U", "S", "D") + helperF(z, "F", "S", "B");
}

Sector::~Sector() {
    // TODO: Free any dynamically allocated memory if necessary
}

Sector& Sector::operator=(const Sector& other) {
    // TODO: Overload the assignment operator
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    this->distance_from_earth = other.distance_from_earth;
    this->sector_code = other.sector_code;
    this->left = other.left;
    this->right = other.right;
    this->parent = other.parent;    
    this->color = other.color;
    return *this;
}
bool Sector::operator<(const Sector& other) const {
    int num = 4*(x > other.x)+ 2*(y > other.y)+ 1*(z > other.z);
    int num2 = 4*(other.x > x)+ 2*(other.y > y)+ 1*(other.z > z);
    return num < num2;
}

bool Sector::operator==(const Sector& other) const {
    return (x == other.x && y == other.y && z == other.z);
}

bool Sector::operator!=(const Sector& other) const {
    return !(*this == other);
}
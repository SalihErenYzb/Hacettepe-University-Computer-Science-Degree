#ifndef PA2_GAMECONTROLLER_H
#define PA2_GAMECONTROLLER_H

#include "BlockFall.h"
#include <functional>
#include <map>
using namespace std;

class GameController {
public:
    bool play(BlockFall &game, const string &commands_file); // Function that implements the gameplay
    void print_grid(BlockFall &game,bool deneme=false); // Prints the grid
    void print_grid_dull(BlockFall &game,bool isShape); // Prints the grid without score and power-up
    void print_grid_dull(BlockFall &game); // Prints the grid without score and power-up
    void rotate_Right(BlockFall &game); // Rotates the active block to the right
    void rotate_Left(BlockFall &game); // Rotates the active block to the left
    void move_Left(BlockFall &game); // Moves the active block to the left
    void move_Right(BlockFall &game); // Moves the active block to the right
    int howMuchDown(BlockFall &game,int x1 , int y1); // Calculates how much the active block's[x1][y1] can go down
    int howMuchDownForShape(BlockFall &game); // Calculates how much the active block can go down
    int dropForGravity(BlockFall &game); // Drops the active block for gravity mode
    int dropForNormal(BlockFall &game); // Drops the active block for normal mode
    bool powerDetection(BlockFall &game); // Checks if the power-up is activated
    int count (BlockFall &game); // Counts the number of blocks in the grid
    int clear(BlockFall &game); // Clears the grid
    int clearRow(BlockFall &game, int row); // Clears the given row
    bool checkIfRowIsFull(BlockFall &game, int row); // Checks if the given row is full
    void drop(BlockFall &game); // Drops the active block
    void power_up(BlockFall &game); // Activates the power-up
    void gravitySwitch(BlockFall &game); // Switches the gravity mode
    bool checkIfCollision(int y, BlockFall &game); // Checks if there is a collision at the given coordinates



};


#endif //PA2_GAMECONTROLLER_H

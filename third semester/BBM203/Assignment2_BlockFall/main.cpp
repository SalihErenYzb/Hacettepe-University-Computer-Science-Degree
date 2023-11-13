#include "BlockFall.h"
#include "GameController.h"
#include <iostream>
#include <iomanip>

void centerText() {


    std::string text =R"(
▀█████████▄   ▄█        ▄██████▄   ▄████████    ▄█   ▄█▄    ▄████████    ▄████████  ▄█        ▄█       
  ███    ███ ███       ███    ███ ███    ███   ███ ▄███▀   ███    ███   ███    ███ ███       ███       
  ███    ███ ███       ███    ███ ███    █▀    ███▐██▀     ███    █▀    ███    ███ ███       ███       
 ▄███▄▄▄██▀  ███       ███    ███ ███         ▄█████▀     ▄███▄▄▄       ███    ███ ███       ███       
▀▀███▀▀▀██▄  ███       ███    ███ ███        ▀▀█████▄    ▀▀███▀▀▀     ▀███████████ ███       ███       
  ███    ██▄ ███       ███    ███ ███    █▄    ███▐██▄     ███          ███    ███ ███       ███       
  ███    ███ ███▌    ▄ ███    ███ ███    ███   ███ ▀███▄   ███          ███    ███ ███▌    ▄ ███▌    ▄ 
▄█████████▀  █████▄▄██  ▀██████▀  ████████▀    ███   ▀█▀   ███          ███    █▀  █████▄▄██ █████▄▄██ 
             ▀                                 ▀                                   ▀         ▀         
)"; // The text to be centered


    std::cout  << text << std::endl;
    cout << endl;
    cout << endl;
    cout << "Welcome to BlockFall!" << endl;
    cout << "Type Start to begin the game!" << endl;
}

int main(int argc, char **argv) {
    std::cout << "\033[2J\033[1;1H";

    // Create a BlockFall instance
    string gravity_mode(argv[4]);

    BlockFall game(argv[1], argv[2], (gravity_mode == "GRAVITY_ON"), argv[5], argv[6]);

    // Create a GameController instance
    GameController controller;
    centerText();
    string abc;
    if (cin >> abc){
    // Play
    controller.play(game, argv[3]);
    }

    return 0;
}
#include "BlockFall.h"
#include "GameController.h"
#include <iostream>
#include <iomanip>
#include "cmdlib.h"
void centerText() {

    std::string string1 = "▀█████████▄   ▄█        ▄██████▄   ▄████████    ▄█   ▄█▄    ▄████████    ▄████████  ▄█        ▄█";
    std::string string2 = "  ███    ███ ███       ███    ███ ███    ███   ███ ▄███▀   ███    ███   ███    ███ ███       ███";
    std::string string3 = "  ███    ███ ███       ███    ███ ███    █▀    ███▐██▀     ███    █▀    ███    ███ ███       ███";
    std::string string4 = " ▄███▄▄▄██▀  ███       ███    ███ ███         ▄█████▀     ▄███▄▄▄       ███    ███ ███       ███";
    std::string string5 = "▀▀███▀▀▀██▄  ███       ███    ███ ███        ▀▀█████▄    ▀▀███▀▀▀     ▀███████████ ███       ███";
    std::string string6 = "  ███    ██▄ ███       ███    ███ ███    █▄    ███▐██▄     ███          ███    ███ ███       ███";
    std::string string7 = "  ███    ███ ███▌    ▄ ███    ███ ███    ███   ███ ▀███▄   ███          ███    ███ ███▌    ▄ ███▌    ▄";
    std::string string8 = " ▄█████████▀ █████▄▄██  ▀██████▀  ████████▀    ███   ▀█▀   ███          ███    █▀  █████▄▄██ █████▄▄██";
    std::string string9 = " ▀                                 ▀                                   ▀         ▀"; // The text to be centered
    int tmp = 93;
    printn(10);
    printt(string1,tmp,true);
        printt(string2,tmp,true);
    printt(string3,tmp,true);
    printt(string4,tmp,true);
    printt(string5,tmp,true);
    printt(string6,tmp,true);
    printt(string7,tmp,true);
    printt(string8,tmp,true);
    printt(string9,tmp,true);
    

    cout << endl;
    cout << endl;
    printt("Welcome to BlockFall!",21,true);
    printt("Type Start to begin the game!",29,true);
    goToMidX(5);
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
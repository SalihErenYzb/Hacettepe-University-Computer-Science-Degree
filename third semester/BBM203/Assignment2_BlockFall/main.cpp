#include "BlockFall.h"
#include "GameController.h"
#include <iostream>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
void print(string text,int pad,bool endline = false){
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    int padding = (w.ws_col - pad) / 2;

    std::cout << std::string(padding, ' ') << text ;
    if (endline){
        cout << endl;
    }

}
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

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    print(text,text.size());

    cout << endl;
    cout << endl;
    print("Welcome to BlockFall!",21,true);
    print("Type Start to begin the game!",29,true);
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
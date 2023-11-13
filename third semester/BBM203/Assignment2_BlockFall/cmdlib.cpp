
#include "cmdlib.h"
using namespace std;
void printt(string text,int pad,bool endline ){
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    int padding = (w.ws_col - pad) / 2;

    // Debugging output

    std::cout << std::string(padding, ' ') << text ;
    if (endline){
        cout << endl;
    }
}
void clearScreen(){
    std::cout << "\033[2J\033[1;1H";
}
void printn(int a){
    for (int i = 0; i < a; i++){
        cout << endl;
    }
}
void goToMidY(int y ,int dividend){
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int padding = (w.ws_row - y) / dividend;
    printn(padding);
}
void goToMidX(int x ,int dividend ){
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int padding = (w.ws_col-x ) / dividend;
    std::cout << std::string(padding, ' ');

}

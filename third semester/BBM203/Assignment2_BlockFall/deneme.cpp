#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    std::string text = "Middle Text";
    int padding = (w.ws_col - text.length()) / 2;

    std::cout << std::string(padding, ' ') << text << std::endl;

    return 0;
}
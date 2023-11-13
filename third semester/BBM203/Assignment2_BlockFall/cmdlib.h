#ifndef CMD_LIB_H
#define CMD_LIB_H

#include <sys/ioctl.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
using namespace std;

void printt(string text,int pad,bool endline = false);
void clearScreen();
void printn(int a);
void goToMidY(int y = 0,int dividend=2);
void goToMidX(int x = 0,int dividend = 2);

#endif
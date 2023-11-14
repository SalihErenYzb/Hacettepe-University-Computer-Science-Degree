#include <iostream>
#include <unistd.h>
struct ha{
    int value;
    ha* next;
};
int main() {
   ha *head = new ha;
   head->next = nullptr;
   head = head->next;
   
}
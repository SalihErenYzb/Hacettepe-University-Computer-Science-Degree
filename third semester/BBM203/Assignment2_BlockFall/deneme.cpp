#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> abc ={15, 12, 10};
    for (size_t i = 0; i < abc.size(); ++i)
    {
        cout << abc[i] << endl;
    }
    return 0;
}
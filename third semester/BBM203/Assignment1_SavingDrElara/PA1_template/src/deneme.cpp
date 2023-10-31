#include "ImageMatrix.h"
#include "Convolution.h"
#include "ImageSharpening.h"
#include "EdgeDetector.h"
#include "DecodeMessage.h"
#include "EncodeMessage.h"
#include "ImageProcessor.h"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int main(){
    double number = 14.999;
    int ha = number;
    cout << ha << endl;
    double k = 1.0;
    ImageMatrix image("output.txt");
    image.print_data();
    ImageSharpening imageSharpening;
    ImageMatrix sharpened = imageSharpening.sharpen(image, 50);
    EdgeDetector edgeDetector;
    cout << "Edge Detector" << endl;
    vector<pair<int,int>> edgePixels = edgeDetector.detectEdges(sharpened);
    //std::sort(edgePixels.begin(), edgePixels.end() );
    std::string message = "";
    int tmp ;
    int lsb;
    for (int i = 0; i < edgePixels.size(); i++) {
        tmp = image.get_data()[edgePixels[i].first][edgePixels[i].second] ;
        lsb = tmp & 1 ;
        message += std::to_string(lsb) ;
    }
    int size = message.size() ;
    if (size% 7 != 0) {
        message += std::string(7-(size % 7), '0') ;
    }
    int size2 = message.size() ;
    cout << "size: " << size2 << endl;
    cout << message << endl;
        std::string endmessage = "" ;
    for (int i = 0; i < size; i += 7) {
        std::string tmp = message.substr(i, 7) ;
        //cout << tmp << endl;
        int number = std::stoi(tmp, nullptr, 2) ;
        //cout << number << endl;
        if (number <= 32 ) {
            number += 33 ;
        }else if (number >= 127) {
            number = 126 ;
        }
        char c = (char)number ;
        endmessage += c ;
    }
    cout << endmessage << endl;
}

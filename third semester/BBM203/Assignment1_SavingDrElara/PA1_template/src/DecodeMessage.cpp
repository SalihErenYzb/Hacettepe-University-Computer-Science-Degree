// DecodeMessage.cpp

#include "DecodeMessage.h"
#include <algorithm>
#include <iostream>
// Default constructor
DecodeMessage::DecodeMessage() {
    // Nothing specific to initialize here
}

// Destructor
DecodeMessage::~DecodeMessage() {
    // Nothing specific to clean up
}


std::string DecodeMessage::decodeFromImage(const ImageMatrix& image, const std::vector<std::pair<int, int>>& edgePixels) {
    std::string message = "";
    //sort edge pixels by row and column
    std::vector<std::pair<int,int>> edgePixels2 = edgePixels ;

    int tmp ;
    int lsb;
    for (int i = 0; i < edgePixels2.size(); i++) {
        tmp = image.get_data()[edgePixels2[i].first][edgePixels2[i].second] ;
        lsb = tmp%2 ;
        message += std::to_string(lsb) ;
    }
    int size3 = message.size() ;
    if (size3% 7 != 0) {
        message = std::string(7-(size3 % 7), '0') + message;
    }
    int size2 = message.size() ;

    std::string endmessage = "" ;
    for (int i = 0; i < size2; i += 7) {
        std::string tmp = message.substr(i, 7) ;
        int number = std::stoi(tmp, nullptr, 2) ;
        if (number <= 32 ) {
            number += 33 ;
        }else if (number >= 127) {
            number = 126 ;
        }        char c = (char)number ;
        endmessage += c ;
    }
    return endmessage ;

}


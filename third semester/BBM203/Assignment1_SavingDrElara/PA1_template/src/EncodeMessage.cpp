#include "EncodeMessage.h"
#include <cmath>



// Default Constructor
EncodeMessage::EncodeMessage() {

}

// Destructor
EncodeMessage::~EncodeMessage() {
    
}

// Function to encode a message into an image matrix
ImageMatrix EncodeMessage::encodeMessageToImage(const ImageMatrix &img, const std::string &message, const std::vector<std::pair<int, int>>& positions) {
    //iterate through the message
    std::string encoded_message = "";
    for (int i = 0; i < message.size(); i++)
    {
        char tmp = message[i];
        tmp += fibonacci(i);
        if (tmp <= 32){
            tmp += 32;
        }
        else if (tmp >= 127){
            tmp  = 126;
        }
        encoded_message += tmp;
    }
    int length = encoded_message.length();
    int shift_amount = length / 2;

    std::string shifted_message = encoded_message.substr(length - shift_amount) + encoded_message.substr(0, length - shift_amount);
    std::string binaryString;

    for (char c : shifted_message) {
        unsigned char charValue = static_cast<unsigned char>(c);
        std::string charBinary;

        for (int i = 6; i >= 0; --i) {
            charBinary += ((charValue & (1 << i)) ? '1' : '0');
        }

        binaryString += charBinary;
    }
    int i = 0;
    ImageMatrix encoded_image(img);

    for (std::pair<int,int> pairr: positions){
        
        int x = pairr.first;
        int y = pairr.second;
        if (binaryString[i] == '1'){
            int tmp2 = encoded_image.get_data()[x][y];
            tmp2 |= 1;
            encoded_image.get_data()[x][y] = tmp2;
        }
        else{
            int tmp3 = encoded_image.get_data()[x][y];
            tmp3 |= 0;
        }

        i++;
    }
    return encoded_image;
}
int EncodeMessage::fibonacci(int n) {
    if (n <= 1) {
        return n;
    }

    int prev = 0;
    int curr = 1;

    for (int i = 2; i <= n; ++i) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }

    return curr;
}

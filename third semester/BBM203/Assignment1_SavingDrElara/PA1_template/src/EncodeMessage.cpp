#include "EncodeMessage.h"
#include <cmath>

#include <iostream>//del later

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
        int  tmp = static_cast<int>( message[i]);
        std::cout << message[i] ;
        if (isPrime(i)){
            tmp = tmp+fibonacci(i);
            if (tmp >= 127){
                tmp  = 126;
            }
            if (tmp <= 32){
                tmp += 33;
            }
        }
    
 
        char tmp2 =  static_cast<char>(tmp);
        encoded_message += tmp2;

    }
    int shiftCount = encoded_message.length() / 2;
    std::string shifted_message = encoded_message;

    for (int i = 0; i < encoded_message.length(); i++) {
        int newIndex = (i + shiftCount) % encoded_message.length();
        shifted_message[newIndex] = encoded_message[i];
    }
    std::string binaryString;
    std::cout << shifted_message << std::endl;  

    for (char c : shifted_message) {
        char charValue = c;
        int charint = static_cast<int>(charValue);
        std::string charBinary;

        while (charint > 0) {
            charBinary.insert(charBinary.begin(), '0' + (charint & 1));
            charint >>= 1;
        }

        while (charBinary.size() < 7) {
            charBinary.insert(charBinary.begin(), '0');
        }
        binaryString += charBinary;
    }

    int i = 0;
    ImageMatrix encoded_image(img);
    std::cout << binaryString << std::endl;
    // print len of positions and binaryString
    std::cout << "size" << positions.size() << std::endl;
    std::cout << "binarysize" << binaryString.size() << std::endl;
    for (std::pair<int,int> pairr: positions){
        if (i >= binaryString.size()){
            break;
        }
        int x = pairr.first;
        int y = pairr.second;
        double useless;
        if (binaryString[i] == '1'){
            int tmp2 = encoded_image.get_data()[x][y];
            if (tmp2 % 2 == 0){
                tmp2 += 1;
            }
            encoded_image.get_data()[x][y] = tmp2;//change
        }
        else{
            int tmp3 = encoded_image.get_data()[x][y];
            if (tmp3 % 2 == 1){
                tmp3 -= 1;
            }
            encoded_image.get_data()[x][y] = tmp3;;
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
bool EncodeMessage::isPrime(int number) {
    if (number <= 1) {
        return false;
    }


    for (int i = 2; i < number; ++i) {
        if (number % i == 0) {
            return false;
        }
    }

    return true;
}
#include "ImageProcessor.h"
#include "ImageMatrix.h"
#include "ImageLoader.h"
#include "Convolution.h"
#include "ImageSharpening.h"
#include "EdgeDetector.h"
#include "DecodeMessage.h"
#include "EncodeMessage.h"
#include <iostream>

ImageProcessor::ImageProcessor() {

}

ImageProcessor::~ImageProcessor() {

}


std::string ImageProcessor::decodeHiddenMessage(const ImageMatrix &img) {
    double k = 2.0;//WTF  CHANGE THIS
    ImageSharpening sharpening;
    ImageMatrix sharpened = sharpening.sharpen(img,k);
    EdgeDetector edge_detector;
    std::vector<std::pair<int, int>> edge_pixels = edge_detector.detectEdges(sharpened);
    DecodeMessage decoder;
    std::string message = decoder.decodeFromImage(img, edge_pixels);
    return message;
}

ImageMatrix ImageProcessor::encodeHiddenMessage(const ImageMatrix &img, const std::string &message) {
    double k = 2.0;//WTF  CHANGE THIS
    ImageSharpening sharpening;
    ImageMatrix sharpened = sharpening.sharpen(img,k);
    EdgeDetector edge_detector;
    std::vector<std::pair<int, int>> edge_pixels = edge_detector.detectEdges(sharpened);
    EncodeMessage encoder;
    ImageMatrix encoded_image = encoder.encodeMessageToImage(img, message, edge_pixels);
    return encoded_image;
}

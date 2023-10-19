#include <iostream>

#include "Convolution.h"

// Default constructor 
Convolution::Convolution() {
}

// Parametrized constructor for custom kernel and other parameters
Convolution::Convolution(double** customKernel, int kh, int kw, int stride_val, bool pad){
}

// Destructor
Convolution::~Convolution() {
}

// Copy constructor
Convolution::Convolution(const Convolution &other){

}

// Copy assignment operator
Convolution& Convolution::operator=(const Convolution &other) {
}


// Convolve Function: Responsible for convolving the input image with a kernel and return the convolved image.
ImageMatrix Convolution::convolve(const ImageMatrix& input_image) const {

}

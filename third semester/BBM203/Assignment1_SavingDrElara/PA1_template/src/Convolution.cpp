#include <iostream>

#include "Convolution.h"

// Default constructor 
Convolution::Convolution() {
    // Default constructor implementation
    kernel = nullptr;
    height = 0;
    width = 0;
    stride = 1;
    padding = false;
}
// Parametrized constructor for custom kernel and other parameters
Convolution::Convolution(double** customKernel, int kh, int kw, int stride_val, bool pad){
    height = kh;
    width = kw;
    stride = stride_val;
    padding = pad;
    kernel = new double*[height];
    for (int i = 0; i < height; i++)
    {
        kernel[i] = new double[width];
        for (int j = 0; j < width; j++)
        {
            kernel[i][j] = customKernel[i][j]; 
        }
        
    }

}

// Destructor
Convolution::~Convolution() {
    if (kernel!=nullptr){
        for (int i = 0; i < height; i++)
        {
            delete[] kernel[i];
        }
        delete[] kernel;
    }
}

// Copy constructor
Convolution::Convolution(const Convolution &other){
    height = other.height;
    width = other.width;
    stride = other.stride;
    padding = other.padding;
    kernel = new double*[height];
    for (int i = 0; i < height; i++)
    {
        kernel[i] = new double[width];
        for (int j = 0; j < width; j++)
        {
            kernel[i][j] = other.kernel[i][j]; 
        }
        
    }
}

// Copy assignment operator
Convolution& Convolution::operator=(const Convolution &other) {
    if (this == &other) {
        return *this; // self-assignment check
    }

    // Deallocate old memory
    if (kernel != nullptr) {
        for (int i = 0; i < height; ++i) {
            delete[] kernel[i];
        }
        delete[] kernel;
    }
    //copy from other
    height = other.height;
    width = other.width;
    stride = other.stride;
    padding = other.padding;
    kernel = new double*[height];
    for (int i = 0; i < height; i++)
    {
        kernel[i] = new double[width];
        for (int j = 0; j < width; j++)
        {
            kernel[i][j] = other.kernel[i][j]; 
        }
        
    }
    return *this;

}


// Convolve Function: Responsible for convolving the input image with a kernel and return the convolved image.
ImageMatrix Convolution::convolve(const ImageMatrix& input_image) const {
    // Convolve function implementation
    int inputHeight = input_image.get_height();
    int inputWidth = input_image.get_width();
    int outputHeight = (inputHeight - height + (padding ? 2 : 0)) / stride + 1;
    int outputWidth = (inputWidth - width + (padding ? 2 : 0)) / stride + 1;
    
    // Create an output image matrix with the appropriate dimensions
    ImageMatrix output_image(outputHeight, outputWidth);
    
    // Perform convolution
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            double sum = 0.0;
            
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int inputRow = i * stride - (padding ? 1 : 0) + k;
                    int inputCol = j * stride - (padding ? 1 : 0) + l;
                    
                    // Apply zero padding if enabled
                    if (padding && (inputRow < 0 || inputRow >= inputHeight || inputCol < 0 || inputCol >= inputWidth)) {
                        sum += 0.0; // Assuming zero padding
                    } else {
                        sum += input_image.get_data()[inputRow][inputCol] * kernel[k][l];
                    }
                }
            }
            
            output_image.get_data()[i][j] = sum;
        }
    }
    
    return output_image;
}
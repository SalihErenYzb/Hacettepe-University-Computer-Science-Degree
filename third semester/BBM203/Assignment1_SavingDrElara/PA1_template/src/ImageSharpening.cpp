#include "ImageSharpening.h"

// Default constructor
ImageSharpening::ImageSharpening() {
    kernel_height = 3;
    kernel_width = 3;
    blurring_kernel = new double*[3];
    for (int i = 0; i < 3; i++)
    {
        blurring_kernel[i] = new double[3];
        for (int j = 0; j < 3; j++)
        {
            blurring_kernel[i][j] = 1.0; 
        }
    }

}

ImageSharpening::~ImageSharpening(){
    for (int i = 0; i < kernel_height; i++)
    {
        delete[] blurring_kernel[i];
    }
    delete[] blurring_kernel;
}

ImageMatrix ImageSharpening::sharpen(const ImageMatrix& input_image, double k) {
    Convolution blurring(blurring_kernel,kernel_height,kernel_width,1,true);
    ImageMatrix blurred = blurring.convolve(input_image);
    blurred = blurred *(1.0/9.0);
    ImageMatrix sharpened = input_image +(input_image - blurred )*k ;
    return sharpened;
}

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
            blurring_kernel[i][j] = (1.0/9.0); 
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

    ImageMatrix sharpened = input_image +(input_image - blurred )*k ;
    /*Clipping: Ensure that pixel intensity values in Isharp lie within acceptable bounds
- [0, 255]. The multiplication with the sharpening factor, k, may cause overflow.

Values that fall outside this range should be clipped to ensure visual coher-
ence, ie, clip the values greater than 255 back to 255.
*/
    for (int i = 0; i < input_image.get_height(); i++)
    {
        for (int j = 0; j < input_image.get_width(); j++)
        {
            sharpened.get_data()[i][j] = std::min(255.0,std::max(0.0,sharpened.get_data()[i][j]));
        }
        
    }
    
    return sharpened;
}

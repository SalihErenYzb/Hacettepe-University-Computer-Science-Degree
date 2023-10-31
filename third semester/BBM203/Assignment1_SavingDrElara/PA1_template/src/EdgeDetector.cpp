// EdgeDetector.cpp

#include "EdgeDetector.h"
#include <cmath>

#include "EdgeDetector.h"
#include <cmath>

// Default constructor
EdgeDetector::EdgeDetector() {
    sobel_x = new double*[3];
    sobel_y = new double*[3];
    for (int i = 0; i < 3; i++)
    {
        sobel_x[i] = new double[3];
        sobel_y[i] = new double[3];
        
    }
    sobel_x[0][0] = -1;
    sobel_x[0][1] = 0;
    sobel_x[0][2] = 1;
    sobel_x[1][0] = -2;
    sobel_x[1][1] = 0;
    sobel_x[1][2] = 2;
    sobel_x[2][0] = -1;
    sobel_x[2][1] = 0;
    sobel_x[2][2] = 1;

    sobel_y[0][0] = -1;
    sobel_y[0][1] = -2;
    sobel_y[0][2] = -1;
    sobel_y[1][0] = 0;
    sobel_y[1][1] = 0;
    sobel_y[1][2] = 0;
    sobel_y[2][0] = 1;
    sobel_y[2][1] = 2;
    sobel_y[2][2] = 1;

    
}

// Destructor
EdgeDetector::~EdgeDetector() {
    for (int i = 0; i < 3; i++)
    {
        delete[] sobel_x[i];
        delete[] sobel_y[i];
    }
    delete[] sobel_x;
    delete[] sobel_y;
}

// Detect Edges using the given algorithm
std::vector<std::pair<int, int>> EdgeDetector::detectEdges(const ImageMatrix& input_image) {
    Convolution convx(sobel_x,3,3,1,true);
    Convolution convy(sobel_y,3,3,1,true);
    ImageMatrix x = convx.convolve(input_image);
    ImageMatrix y = convy.convolve(input_image);
    //Gradient magnitude calculation
    ImageMatrix mag(x.get_height(),x.get_width());
    double sum = 0;
    double tmp;
    for (int i = 0; i < x.get_height(); i++)
    {
        for (int j = 0; j < x.get_width(); j++)
        {
            tmp = sqrt(pow(x.get_data()[i][j],2)+pow(y.get_data()[i][j],2));
            mag.get_data()[i][j] = tmp;
            sum+=tmp;
        }
        
    }
    double avg = sum/(x.get_height()*x.get_width());
    //Filling edge vector
    std::vector<std::pair<int, int>> edge_vector;
    for (int i = 0; i < x.get_height(); i++)
    {
        for (int j = 0; j < x.get_width(); j++)
        {
            if(mag.get_data()[i][j]>avg){
                edge_vector.push_back(std::make_pair(i,j));
            }
        }
        
    }
    return edge_vector;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float sigmoidf(float x){
    return 1.f / ( 1.f + expf(-x) ) ;
}

// XOR-> can't be modelable by single neuron-+

// Dataset
// OR Gate
// float train[][3] = {
//     // x,y,o/p
//     {0, 0, 0},
//     {0, 1, 1},
//     {1, 0, 1},
//     {1, 1, 1},
// };
//AND
float train[][3] = {
    // x,y,o/p
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};
// XOR can't be modeled by a single neuron
// float train[][3] = {
//     // x,y,o/p
//     {0, 0, 0},
//     {0, 1, 1},
//     {1, 0, 1},
//     {1, 1, 0},
// };



// Macro to calculate the number of rows in the dataset
#define train_count (sizeof(train) / sizeof(train[0]))

//random float between 0 and 1
float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

// cost function
float cost_function(float w1,float w2,float b){
    // Iterate through the dataset
    float result = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = x1*w1 + x2*w2  +b;
        // printf("Y; %f ,",y);
        y = sigmoidf(y) ;
        // printf("Y_S; %f \n",y);
        // printf("X1: %0.2f, X2: %0.2f, Y_E: %0.2f, Y_P: %0.2f\n", x1, x2, train[i][2], y);
        float d = y - train[i][2];
        result+= d*d;
    }
    return result/train_count;
}
// y = w * x (model)
int main() {

    srand(time(0));
    float w1= rand_float();
    float w2= rand_float();
    float b= rand_float();
    printf("Initial W: w1:%f ,w2: %f, b:%f\n",w1,w2,b);

    float c=cost_function(w1,w2,b);
    printf("CF: %f\n",c);

    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            printf("%zu | %zu : %f | Y:%f\n",i,j,sigmoidf(i*w1 + j*w2 + b),i*w1 + j*w2 + b);
        }
    }

    float esp = 1e-2; 
    float rate = 1e-3;
    for(int i=0;i<1000000;i++){
        c=cost_function(w1,w2,b);
        float dw1 = ( cost_function(w1+esp ,w2,b) - c ) / esp;  // grad
        float dw2 = ( cost_function(w1 ,w2+esp ,b) - c ) / esp; // grad
        float db = ( cost_function(w1 ,w2 ,b+esp) - c ) / esp; // grad
        w1-= rate*dw1; 
        w2-= rate*dw2;
        b-= rate*db;
    }
    printf("Updated: w1:%f ,w2:%f ,b:%f\n",w1,w2,b);
    printf("CF: %f\n",cost_function(w1,w2,b));

// The weight is increasing very slow so we come up with Activation Fnction i.e SigmoidFunction
    // sigmoid
    // for(float i=-10.0f;i<=10.0f;i++){
    //     printf("X: %f ,S: %f\n ",i,sigmoidf(i));
    // }

    // checking model performance
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            printf("%zu | %zu : %f | Y:%f\n",i,j,sigmoidf(i*w1 + j*w2 + b),i*w1 + j*w2 + b);
        }
    }
    return 0;
}

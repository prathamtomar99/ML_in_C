#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    // actual, predicted
    {0, 0},
    {1, 2},
    {2, 4},
    {3,6},
    {4, 8}, 
};

// Macro to calculate the number of rows in the dataset
#define train_count (sizeof(train) / sizeof(train[0]))

//random float between 0 and 1
float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

// cost function
float cost_function(float w,float b){
    // Iterate through the dataset
    float result = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float x = train[i][0];
        float y = x*w+b;
        // printf("Actual: = %0.2f, Expected = %0.2f\n", y, train[i][1]);
        float d = y - train[i][1];
        result+=d*d;
    }
    return result/train_count;
}

// y = w * x (model) 

int main() {
    // srand(time(0)); // Seed the random number generator
    srand(69);

//random initial weight
    float w_rand = rand_float() * 10000.00f;
    float bias =  rand_float() * 5000.00f;
    // printf("Random Weight: %0.4f\n", w_rand);

// cose function
    float loss = cost_function(w_rand,bias);
    
    // printf("CostFunction %f\n",loss);

// minimize cost function
    // 1e-1 = 0.1
    float h=1e-1; //eps
    float rate=1e-1;
    printf("C: %f, W: %f, B: %f .\n",cost_function(w_rand,bias),w_rand,bias);
    for(int i=0;i<101;i++){
        float c = cost_function(w_rand,bias);
        float dw = (cost_function(w_rand + h,bias)-c) /h;
        float db = (cost_function(w_rand ,h + bias)-c) /h;
        bias -= rate*db;
        w_rand -= rate * dw;
        // printf("C: %f, W: %f, B: %f .\n",c,w_rand,bias);
    }
    printf("C: %f, W: %f, B: %f .\n",cost_function(w_rand,bias),w_rand,bias);
    // printf("loss: Before %0.2f,\nAfter %0.2f\n",loss,dcost);
    return 0;
}

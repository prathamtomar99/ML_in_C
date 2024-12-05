#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

// making training dataset for AND and OR
float train_NAND[][3] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};
float train_OR[][3] = {
    {0,0,0,},
    {0,1,1},
    {1,0,1},
    {1,1,1},
};
float train_AND[][3] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1},
};

// size for each train_dataset
#define size_AND sizeof(train_AND)/sizeof(train_AND[0])
#define size_OR sizeof(train_OR)/sizeof(train_OR[0])
#define size_NAND sizeof(train_NAND)/sizeof(train_NAND[0])

// create a function to assign random values
float random_f(void){
    return (float) rand()/ (float) RAND_MAX ; 
}

// Sigmoid (for activation function)
float sigmoidf(float x){
    return 1/ (1 + expf(-x));
}

// cost function NAND
float cost_function_NAND(float w1, float w2,  float b){
    float result = 0.0f;
    for(size_t i = 0 ; i < size_NAND; ++i ){
        float predicted = sigmoidf ( w1*train_NAND[i][0] + w2*train_NAND[i][1] + b) ;
        float actual = train_NAND[i][2];
        float d = predicted-actual;
        d=d*d;
        result+=d;
    }
    return result / size_NAND;
}
// cost function AND
float cost_function_AND(float w1, float w2,  float b){
    float result = 0.0f;
    for(size_t i = 0 ; i < size_AND; ++i ){
        float predicted = sigmoidf ( w1*train_AND[i][0] + w2*train_AND[i][1] + b) ;
        float actual = train_AND[i][2];
        float d = predicted-actual;
        d=d*d;
        result+=d;
    }
    return result / size_AND;
}
// cost function OR
float cost_function_OR(float w1, float w2,float b){
    float result = 0.0f;
    for(size_t i = 0 ; i < size_OR; ++i){
        float predicted = sigmoidf ( w1*train_OR[i][0] + w2*train_OR[i][1] + b) ;
        float actual = train_OR[i][2];
        float d = predicted-actual;
        d=d*d;
        result+=d;
    }
    return result / size_OR;
}

int main(){
    srand(time(0));

    // Random weights
    float w1 = random_f(); 
    float w2 = random_f();
    float w3 = random_f();
    float w4 = random_f();
    float w5 = random_f();
    float w6 = random_f();

    // Random bias
    float b1 = random_f();
    float b2 = random_f();
    float b3 = random_f();

    printf("-------------------------------------\n");
    printf("Random Initial W&B \nNAND: W1:%0.4f  W2:%0.4f  B1:%0.4f\nOR: W3:%0.4f  W4:%0.4f  B2:%0.4f\nAND: W5:%0.4f  W6:%0.4f B3:%0.4f\n",w1,w2,b1,w3,w4,b2,w5,w6,b3);

    // // check sigmoid working
    // for(float i = -10.f;i<11.f;i++){
    //     printf("%f : %f\n",i,sigmoidf(i));
    // }


    // define LRate and exp (h in derivation)
    float rate =1e-1;
    float exp = 1e-1;
// adjust weight and bias using cost function
    // AND
    printf("-------------------------------------\n");
    float cf = cost_function_NAND(w1,w2,b1);
    printf("NAND Initial CF: %f\n",cf);
    
    for(int i=0;i<1000*10000;i++){
        float c = cost_function_NAND(w1,w2,b1);
        float dw1 = ( cost_function_NAND(w1+exp,w2,b1) - c )/ exp ;
        float dw2 = ( cost_function_NAND(w1,w2+exp,b1) - c )/ exp;
        float db1 = ( cost_function_NAND(w1,w2,b1+exp) - c )/ exp;
        // updating
        w1 = w1 - (rate * dw1) ;
        w2 = w2 - (rate * dw2) ;
        b1 = b1 - (rate * db1) ;
    }
    cf = cost_function_NAND(w1,w2,b1);
    printf("NAND Final CF: %f\n",cf);
    printf("-------------------------------------\n");

    // OR
    cf = cost_function_OR(w3,w4,b2);
    printf("-------------------------------------\n");
    printf("OR Initial CF: %f\n",cf);
    
    for(int i=0;i<1000*10000;i++){
        float c = cost_function_OR(w3,w4,b2);
        float dw1 = ( cost_function_OR(w3+exp,w4,b2) - c )/ exp ;
        float dw2 = ( cost_function_OR(w3,w4+exp,b2) - c )/ exp;
        float db1 = ( cost_function_OR(w3,w4,b2+exp) - c )/ exp;
        // updating
        w3 -= (rate * dw1) ;
        w4 -= (rate * dw2) ;
        b2 -= (rate * db1) ;
    }
    cf = cost_function_OR(w3,w4,b2);
    printf("OR Final CF: %f\n",cf);
    printf("-------------------------------------\n");

    // AND
    printf("-------------------------------------\n");
    cf = cost_function_AND(w5,w6,b3);
    printf("AND Initial CF: %f\n",cf);
    
    for(int i=0;i<1000*10000;i++){
        float c = cost_function_AND(w5,w6,b3);
        float dw1 = ( cost_function_AND(w5+exp,w6,b3) - c )/ exp ;
        float dw2 = ( cost_function_AND(w5,w6+exp,b3) - c )/ exp;
        float db1 = ( cost_function_AND(w5,w6,b3+exp) - c )/ exp;
        // updating
        w5 -= (rate * dw1) ;
        w6 -= (rate * dw2) ;
        b3 -= (rate * db1) ;
    }
    cf = cost_function_AND(w5,w6,b2);
    printf("AND Final CF: %f\n",cf);
    printf("-------------------------------------\n");


    printf("Random FINAL W&B \n NAND: W1:%0.4f  W2:%0.4f  B1:%0.4f\nOR: W3:%0.4f  W4:%0.4f  B2:%0.4f\nAND: W5:%0.4f  W6:%0.4f B3:%0.4f\n",w1,w2,b1,w3,w4,b2,w5,w6,b3);

    printf("-------------------------------------\n");

// test the weights and bias
    // NAND 0.3
    printf("-------------------------------------\n");
    printf("NAND:\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
                printf("%zu | %zu | %f\n",i,j,sigmoidf(w1*i + w2*j +  b1));
        }
    }
    // OR 0.7
    printf("-------------------------------------\n");
    printf("OR:\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
                printf("%zu | %zu | %f\n",i,j,sigmoidf(w3*i + w4*j  + b2));
            
        }
    }
    // AND 0.3
    printf("-------------------------------------\n");
    printf("AND:\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
                printf("%zu | %zu | %f\n",i,j,sigmoidf(w5*i + w6*j + b3));
        }
    }

    // XOR 0.3
    printf("-------------------------------------\n");
    printf("XOR:\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            for(size_t k =0;k<2;k++){
                float x1 = sigmoidf(w1*i + w2*j + b1);  // NAND
                float x2 = sigmoidf(w3*i + w4*j + b2);  // OR

                float l = sigmoidf(w5*x1 + w6*x2 + b3); // AND

                float x3 = sigmoidf(w1*l + w2*k + b1);  // NAND
                float x4 = sigmoidf(w3*l + w4*k + b2);  // OR
                float x5 = sigmoidf(w5*x3 + w6*x4 + b3); // AND
                printf("%zu | %zu | %zu | %f \n",i,j,k,x5 );
            }
        }
    }

    return 0;
}

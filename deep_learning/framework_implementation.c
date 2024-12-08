#define NN_IMPLEMENTATION
#include "nn.h"
#define mat_rand_default(m) mat_rand((m), 0.0f, 1.0f) // macro for default high and low values
#define MAT_AT(m,i,j) (m).es[(i)*(m).cols + (j)]  // macro to print the 1D array as 2D

typedef struct{
    size_t rows;
    size_t cols;
    float *es;   // element start 
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b);  // dest,a,b
void mat_sum(Mat dst,Mat a);
void mat_print(Mat m); 
void mat_fill(Mat m, float x);
void mat_identity(Mat m);

float rand_float(void){
    return (float) rand()/ (float) RAND_MAX;
}
// define functions
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL); // if condition satisfies then execute the rest program
    return m;
}
void mat_sum(Mat dst,Mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for(size_t i=0;i<dst.rows;i++){
        for(size_t j=0;j<dst.cols;j++){
            MAT_AT(dst,i,j) += MAT_AT(a,i,j);
        }
    }
}
void mat_print(Mat m){
    for(size_t i=0;i<m.rows;i++){
        for(size_t j=0;j<m.cols;j++){
            // printf("%f",m.es[i*m.cols+j]);
            printf("%f ",MAT_AT(m,i,j));
        }
        printf("\n");
    }
}
void mat_rand(Mat m,float low,float high){
    for(size_t i=0;i<m.rows;i++){
        for(size_t j=0;j<m.cols;j++){
            // m.es[i*m.cols+j]=rand_float()*(high-low)+low;
            MAT_AT(m,i,j)=rand_float()*(high-low)+low;
        }
        // printf("\n");
    }
}

void mat_fill(Mat m, float x){
    for(size_t i =0;i<m.rows;i++){
        for(size_t j =0;j<m.cols;j++){
            MAT_AT(m,i,j)=x;
        }
    }
}
void mat_dot(Mat dst, Mat a, Mat b){
    assert(a.cols == b.rows);
    size_t n =a.cols;
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    
    for(size_t i = 0; i<a.rows;i++){
        for(size_t j=0;j<b.cols;j++){
            float sum = 0.f;
            for(size_t k=0;k<a.cols;k++){
                sum+=(MAT_AT(a,i,k)*MAT_AT(b,k,j));
            }
            MAT_AT(dst,i,j)=sum;
        }
    }
}
void mat_identity(Mat m){
    assert(m.rows==m.cols);
    for(size_t i=0;i<m.rows;i++){
        for(size_t j=0;j<m.cols;j++){
            if(i==j)
                MAT_AT(m,i,j)=1;
            else
                MAT_AT(m,i,j)=0;
        }
    }
}

int main(void){
    // double *es;
    // printf("%lu",sizeof(*es));

    Mat m1=mat_alloc(5,3);
    mat_rand(m1,1,5);
    printf("M1:Random values between 1&5:  shape=5X3\n");
    mat_print(m1);
    
    Mat m2=mat_alloc(5,3);
    mat_rand(m2,10,15);
    printf("M2: Random values between 10&15  shape=5X3\n");
    mat_print(m2);

// matrix sum
    mat_sum(m2,m1);
    printf("M3: Matrix Sum M1+M2\n");
    mat_print(m2);

// matrix dot
    Mat m4=mat_alloc(5,3);
    Mat m5=mat_alloc(3,6);
    Mat m6= mat_alloc(5,6);
    mat_rand(m4,1,5);
    printf("M4: Random values between 1&5:  shape=5X3\n");
    mat_print(m4);
    mat_rand(m5,1,3);
    printf("M5: Random values between 1&3:  shape=3X6\n");
    mat_print(m5);
    mat_dot(m6,m4,m5);
    printf("M6: Dot Product M4*M5:  shape=5X6\n");
    mat_print(m6);

// checking the matrix identity property
    Mat m7 = mat_alloc(4,4);
    Mat m8 = mat_alloc(4,4);
    Mat m9 = mat_alloc(4,4);
    mat_rand(m7,1,5);
    mat_identity(m8);
    printf("M7: Random values between 1&5:  shape=4X4\n");
    mat_print(m7);
    printf("M8: Identity Matrix:  shape=4X4 \n");
    mat_print(m8);  
    mat_dot(m9,m7,m8);
    printf("M9: M7 * M8(I) = M7    shape=4X4\n");
    mat_print(m9);
    return 0;
}

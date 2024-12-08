#ifndef NN_FRAMEWORK_H
#define NN_FRAMEWORK_H
#include "nn.h"
#define MAT_AT(m,i,j) (m).es[(i)*(m).stride + (j)]  // macro to print the 1D array as 2D
#define MAT_PRINT(m) mat_print(m,#m,0)

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride; // number of elements per row
    float *es;   // element start 
} Mat;

float rand_float(void);
float sigmoidf(float x);
 
Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b);  // dest,a,b dot(multiplication of a and b)
void mat_sum(Mat dst,Mat a); // dest = dst + a;
void mat_print(Mat m,const char *name, size_t paddding); 
void mat_fill(Mat m, float x); // all elem with x
void mat_identity(Mat m); // all elem with 1
void mat_rand(Mat m,float low,float high);  // all elem within low to high
void mat_sig(Mat m);  // sigmoid of each elem inside mat
Mat mat_row(Mat m,size_t row);  // return one whole row as another mat (subarray)
void mat_copy(Mat dst,Mat a); // dst = a

float sigmoidf(float x){
    return 1.f/(1.f+expf(-x));
}
float rand_float(void){
    return (float) rand()/ (float) RAND_MAX;
}

// define functions
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
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
void mat_print(Mat m,const char *name, size_t paddding){
    printf("%*s%s = [\n", (int) paddding,"",name);
    for(size_t i=0;i<m.rows;i++){
        printf("  %*s", (int) paddding*2,"");
        for(size_t j=0;j<m.cols;j++){
            // printf("%f",m.es[i*m.cols+j]);
            printf("%f ",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("%*s ]\n",(int) paddding,"");
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
void mat_sig(Mat m){
    for(size_t i=0;i<m.rows;i++){
        for(size_t j=0;j<m.cols;j++){
            MAT_AT(m,i,j)=sigmoidf(MAT_AT(m,i,j));
        }
    }
}
Mat mat_row(Mat m,size_t row) {
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m,row,0),
    };
}
void mat_copy(Mat dst,Mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for(size_t i =0 ;i<a.rows ; i++){
        for(size_t j=0; j<a.cols;j++){
            MAT_AT(dst,i,j) = MAT_AT(a,i,j);
        }
    }
}

#define mat_rand_default(m) mat_rand((m), 0.0f, 1.0f) // macro for default high and low values
#endif

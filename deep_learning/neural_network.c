#include "nn_framework.h"
#define array_len(xs) sizeof((xs))/sizeof((xs)[0])
#define NN_PRINT(m) nn_print(m,#m)
#define MAT_PRINT(m) mat_print(m,#m,0)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[((nn).count)]


typedef struct{
    size_t count;
    Mat *as;  // o/p of rach layer arrray
    Mat *ws;  // weight array
    Mat *bs;  // bias arra y
} NN;

// size_t arch[] = {2,2,1};  // i/p layer 1st layer last layer 
// NN nn = nn_alloc(arch,array_len(arch));

NN nn_alloc(size_t *arch, size_t arch_count){

    assert(arch_count>0);
    NN nn;
    nn.count = arch_count-1; // it includes input layer

    nn.ws=malloc(sizeof(*nn.ws)*nn.count);
    assert(nn.ws != NULL);
    nn.bs=malloc(sizeof(*nn.bs)*nn.count);
    assert(nn.bs != NULL);
    nn.as=malloc(sizeof(*nn.as)*(nn.count+1));
    assert(nn.as != NULL); 

    nn.as[0] = mat_alloc(1,arch[0]);
    for(size_t i =1;i<arch_count;i++){
        nn.ws[i-1]= mat_alloc(nn.as[i-1].cols,arch[i]);
        nn.bs[i-1]= mat_alloc(1, arch[i]);
        nn.as[i]  = mat_alloc(1, arch[i]);
    }
    return nn;
}
void nn_print(NN nn, const char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn .count; i++) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4); 
    }
    printf("]\n");
}
void nn_rand(NN nn, float high,float low){
    for(size_t i=0;i<nn.count;i++){
        mat_rand(nn.ws[i],high,low);
        mat_rand(nn.bs[i],high,low);
    }
}
void nn_forward(NN nn){
    for(size_t i=0;i<nn.count;i++){
        mat_dot(nn.as[i+1],nn.as[i],nn.ws[i]);
        mat_sum(nn.as[i+1],nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}
float nn_cost(NN nn, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert (to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;
    float cost = 0.0f;

    for(size_t i=0;i<n;i++){
        Mat x = mat_row(ti,i);  // input
        Mat y = mat_row(to,i);  // output
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for(size_t j=0;j<q;j++){ // matrix is 1X1 as a result it will undergo one loop
            float d = MAT_AT(NN_OUTPUT(nn) ,0,j) - MAT_AT(y,0,j);
            cost += d*d;
        }
    }
    return cost/n;
}
void nn_finite_diff(NN nn,NN g, float eps,Mat ti, Mat to){
    float saved;
    float c = nn_cost(nn,ti,to);

    for(size_t i=0;i<nn.count;i++){
        for(size_t j=0;j<nn.ws[i].rows;j++){
            for(size_t k=0;k<nn.ws[i].cols;k++){
                saved = MAT_AT(nn.ws[i],j,k);
                MAT_AT(nn.ws[i],j,k) += eps;
                MAT_AT(g.ws[i],j,k) = (nn_cost(nn,ti,to) - c)/eps;
                MAT_AT(nn.ws[i],j,k) = saved;
            }
        }

        for(size_t j=0;j<nn.bs[i].rows;j++){
            for(size_t k=0;k<nn.bs[i].cols;k++){
                saved = MAT_AT(nn.bs[i],j,k);
                MAT_AT(nn.bs[i],j,k) += eps;
                MAT_AT(g.bs[i],j,k) = (nn_cost(nn,ti,to) - c)/eps;
                MAT_AT(nn.bs[i],j,k) = saved;
            }
        }
    } 
}
void nn_learn(NN nn,NN g, float rate){
    for(size_t i=0;i<nn.count;i++){
        for(size_t j=0;j<nn.ws[i].rows;j++){
            for(size_t k=0;k<nn.ws[i].cols;k++){
                MAT_AT(nn.ws[i],j,k) -= rate*MAT_AT(g.ws[i],j,k);
            }
        }

        for(size_t j=0;j<nn.bs[i].rows;j++){
            for(size_t k=0;k<nn.bs[i].cols;k++){
                MAT_AT(nn.bs[i],j,k) -= rate*MAT_AT(g.bs[i],j,k);
            }
        }
    }  
}

float td_xor[]={
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
float td_or[]={
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,1,
};

int main(){
    srand(time(0));

    float *td = td_or;
    // float *td = td_xor; // 2,2,1
    float stride = 3;
    size_t n = 4;
    Mat ti ={
        .rows = n,
        .cols =2,
        .stride = stride,
        .es = td,    
    };
    Mat to ={
        .rows = n,
        .cols =1,
        .stride = stride,
        .es = td+2,
    };
    
    size_t arch[] ={2,1}; //or
    // size_t arch[] ={2,2,1}; // xor
    NN nn=nn_alloc(arch , array_len(arch));
    NN g=nn_alloc(arch , array_len(arch));
    nn_rand(nn,0,1);
    // NN_PRINT(nn);

    // MAT_PRINT(mat_row(ti,1 ));
    // Mat row = mat_row(ti,1 );
    // MAT_PRINT(row);
    // mat_copy(NN_INPUT(nn),mat_row(ti,1 ));
    // nn_forward(nn);
    // MAT_PRINT(NN_OUTPUT(nn));

    float eps = 1e-1;
    float rate = 1e-1;

    printf("Cost id = %f\n",nn_cost(nn,ti,to));
    for(int i=0;i<100000;i++){
        nn_finite_diff(nn,g,eps,ti,to);
        nn_learn(nn,g,rate);
    }
    printf("Cost id = %f\n",nn_cost(nn,ti,to));
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            MAT_AT(NN_INPUT(nn),0,0) = i;
            MAT_AT(NN_INPUT(nn),0,1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu : %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0)  ); // only 1 element element 0,0
        }
    }
    return 0;
}

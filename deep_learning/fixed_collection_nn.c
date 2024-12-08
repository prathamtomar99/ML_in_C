#include "framework.h"

typedef struct{
    Mat a0; // input
    Mat w1,a1,b1;  // layer 1
    Mat w2,a2,b2;  // layer 2
} Xor;

Xor xor_alloc(void){
    Xor m;
    // input matrix
    m.a0=mat_alloc(1,2);
    // allocating memory for all weight + bias
    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);
    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a1 = mat_alloc(1,2); // intermidiate store the value of x*w +b (first layer)
    m.a2 = mat_alloc(1,1); // (second layer)
    return m;
}

void forward_xor(Xor m){
// sigmoid (x*w +b)
    // frist layer
    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);
    // second layer
    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);
}

float cost(Xor m,Mat ti, Mat to) {  // inp , out
    assert(ti.rows == to.rows);
    assert (to.cols == m.a2.cols);

    size_t n = ti.rows;
    float cost = 0.0f;

    for(size_t i=0;i<n;i++){
        Mat x = mat_row(ti,i);  // input
        Mat y = mat_row(to,i);  // output
        mat_copy(m.a0, x);
        forward_xor(m);
        size_t q = to.cols;
        for(size_t j=0;j<q;j++){ // matrix is 1X1 as a result it will undergo one loop
            float d = MAT_AT(m.a2,0,j) - MAT_AT(y,0,j);
            cost += d*d;
        }
    }
    return cost/n;
}

void finite_diff(Xor m, Xor g, float eps,Mat ti,Mat to){  // find grad
    float saved;
    float c = cost(m,ti,to);
    for(size_t i=0;i<m.w1.rows;i++){
        for(size_t j=0;j<m.w1.cols;j++){
            saved = MAT_AT(m.w1,i,j);
            MAT_AT(m.w1,i,j) += eps;
            MAT_AT(g.w1,i,j) = (cost(m,ti,to) - c)/eps;
            MAT_AT(m.w1,i,j) = saved;
        }
    }
    for(size_t i=0;i<m.b1.rows;i++){
        for(size_t j=0;j<m.b1.cols;j++){
            saved = MAT_AT(m.b1,i,j);
            MAT_AT(m.b1,i,j) += eps;
            MAT_AT(g.b1,i,j) = (cost(m,ti,to) - c)/eps;
            MAT_AT(m.b1,i,j) = saved;
        }
    }
    for(size_t i=0;i<m.w2.rows;i++){
        for(size_t j=0;j<m.w2.cols;j++){
            saved = MAT_AT(m.w2,i,j);
            MAT_AT(m.w2,i,j) += eps;
            MAT_AT(g.w2,i,j) = (cost(m,ti,to) - c)/eps;
            MAT_AT(m.w2,i,j) = saved;
        }
    }
    for(size_t i=0;i<m.b2.rows;i++){
        for(size_t j=0;j<m.b2.cols;j++){
            saved = MAT_AT(m.b2,i,j);
            MAT_AT(m.b2,i,j) += eps;
            MAT_AT(g.b2,i,j) = (cost(m,ti,to) - c)/eps;
            MAT_AT(m.b2,i,j) = saved;
        }
    }
}
void learn(Xor m, Xor g, float rate){
    for(size_t i=0;i<m.w1.rows;i++){
        for(size_t j =0;j<m.w1.cols;j++){
            MAT_AT(m.w1,i,j)=MAT_AT(m.w1,i,j)-(rate*MAT_AT(g.w1,i,j)); 
        }
    }
    for(size_t i=0;i<m.b1.rows;i++){
        for(size_t j =0;j<m.b1.cols;j++){
            MAT_AT(m.b1,i,j)=MAT_AT(m.b1,i,j)-(rate*MAT_AT(g.b1,i,j)); 
        }
    }
    for(size_t i=0;i<m.w2.rows;i++){
        for(size_t j =0;j<m.w2.cols;j++){
            MAT_AT(m.w2,i,j)=MAT_AT(m.w2,i,j)-(rate*MAT_AT(g.w2,i,j)); 
        }
    }
    for(size_t i=0;i<m.b2.rows;i++){
        for(size_t j =0;j<m.b2.cols;j++){
            MAT_AT(m.b2,i,j)=MAT_AT(m.b2,i,j)-(rate*MAT_AT(g.b2,i,j)); 
        }
    }
}

float td[]={
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main(){
    srand(time(0));

    float stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
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
    // mat_print(ti);
    // mat_print(to);
    // return 0;
    
    Xor m= xor_alloc();
    Xor g= xor_alloc();

    // assigning vale default random between [0,1]
    mat_rand_default(m.w1);
    mat_rand_default(m.w2);
    mat_rand_default(m.b1);
    mat_rand_default(m.b2);


// // print
    // printf("W1: 2X2\n");
    // mat_print(m.w1);
    // printf("B1: 1X2\n");
    // mat_print(m.b1);
    // printf("W2: 2X1\n");
    // mat_print(m.w2);
    // printf("B2: 1X1\n");
    // mat_print(m.b2);

for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            MAT_AT(m.a0,0,0) = i;
            MAT_AT(m.a0,0,1) = j;
            forward_xor(m);
            float y= *m.a2.es;
            // float y=MAT_AT(m.a2,0,0);
            printf("%zu ^ %zu : %f\n",i,j,y  );
        }
    }
    printf("cost = %f\n",cost(m,ti,to));
    for(int i=0;i<1000*1000;i++){
        float eps = 1e-1;
        float rate = 1e-1;
        finite_diff(m,g,eps,ti,to);
        learn(m,g,rate);
        // printf("cost = %f\n",cost(m,ti,to));
    }

    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            MAT_AT(m.a0,0,0) = i;
            MAT_AT(m.a0,0,1) = j;
            forward_xor(m);
            float y= *m.a2.es;
            // float y=MAT_AT(m.a2,0,0);
            printf("%zu ^ %zu : %f\n",i,j,y  );
        }
    }
}

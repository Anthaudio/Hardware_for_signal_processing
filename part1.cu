#include <stdio.h>
#include <stdlib.h>


//on crée une fonction qui initialise une matrice de taille n*p
//avec des valeurs aléatoires comprises entre -1 et 1

void MatrixInit(float *M, int n, int p) {
    float rd_val;
    for (int i=0; i<n*p; i++){
        rd_val = (float)rand()/(float)(RAND_MAX/1.0) - 0.5;
        M[i]=rd_val;
    }   
}


void MatrixPrint(float *M, int n, int p){
    for(int l=0; l<n;l++){
        for(int c=0; c<p; c++){
            printf("%1.1f", M[l*p+c]);
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i=0; i<n*p; i++){
        Mout[i] = M1[i]+M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

}


int main(){
    float *M;
    float *M2;
    float *Mout;

    int n = 3;
    int p = 3;

    M = (float*)malloc(n*p*sizeof(float));
    M2 = (float*)malloc(n*p*sizeof(float));
    Mout = (float*)malloc(n*p*sizeof(float));

    MatrixInit(M,n,p);
    MatrixPrint(M,n,p);
    MatrixInit(M2,n,p);
    MatrixPrint(M2,n,p);

    MatrixAdd(M, M2, Mout, n, p);
    MatrixPrint(Mout,n,p);
}


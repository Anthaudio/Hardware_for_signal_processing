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
    
    printf("Addition from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < n && col < p){
        Mout[lig * p + col] = M1[lig * p + col] + M2[lig * p + col];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
    printf("Multiplication from the CPU...\n\n");
    
    for (int lig = 0; lig < n; lig++){
        for (int col = 0; col < n; col++){
            float s = 0.0f;
            for (int i = 0; i < n; i++) {
                s += M1[lig * n + i] * M2[i * n + col];
            }
            Mout[lig * n + col] = s;
        }
    }
}




int main(){
    //Dans le CPU
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

    MatrixMult(M,M2,Mout,n);
    MatrixPrint(Mout,n,p);

    //dans le GPU
    
    float *d_M, *d_M2, *d_Mout;

    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_M, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * p);

    cudaMemcpy(d_M, M, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    dim3 block_size(n, p);
    dim3 grid_size(1, 1);

    // Addition sur GPU
    cudaMatrixAdd<<<grid_size, block_size>>>(d_M, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();
    
    // Copie du résultat sur CPU
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MatrixPrint(Mout, n, p);
}


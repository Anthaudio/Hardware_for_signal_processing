#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//on crée une fonction qui initialise une matrice de taille n*p
//avec des valeurs aléatoires comprises entre -1 et 1

#define PRINT 0

void MatrixInit(float *M, int n, int p) {
    float rd_val;
    for (int i=0; i<n*p; i++){
        rd_val = (float)rand()/(float)(RAND_MAX/1.0) - 0.5;
        M[i]=rd_val;
    }   
}


void MatrixPrint(float *M, int n, int p){
    if (PRINT == 1){
        return;
    }
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
    int index = threadIdx.x * p +  blockIdx.x;
    Mout[index] = M1[index] + M2[index];
    
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
   
    
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

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p) {
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    if (lig < n && col < n) {
        for (int i = 0; i < p; i++) {
            s += M1[lig * p + i] * M2[i * n + col];
        }
        Mout[lig * n + col] = s;
    }
}


int main(){
    clock_t start_clock_CPU, start_clock_GPU, end_clock_CPU, end_clock_GPU;
    double CPU_time, GPU_time;
    
    //Dans le CPU
    float *M;
    float *M2;
    float *Mout;

    int n = 2;
    int p = 2;

    int N = n*p;

    M = (float*)malloc(n*p*sizeof(float));
    M2 = (float*)malloc(n*p*sizeof(float));
    Mout = (float*)malloc(n*p*sizeof(float));

    MatrixInit(M,n,p);
    //MatrixPrint(M,n,p);
    MatrixInit(M2,n,p);
    //MatrixPrint(M2,n,p);

    start_clock_CPU = clock();
    MatrixAdd(M, M2, Mout, n, p);
    end_clock_CPU = clock();
    //MatrixPrint(Mout,n,p);

    MatrixMult(M,M2,Mout,n);
    //MatrixPrint(Mout,n,p);

    //dans le GPU
    
    float *d_M, *d_M2, *d_Mout;

    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_M, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * p);

    cudaMemcpy(d_M, M, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    dim3 grid_size(n, n, 1);
    dim3 block_size(p, p, 1);
    

    // Addition sur GPU
    cudaMatrixAdd<<<grid_size, block_size>>>(d_M, d_M2, d_Mout, n, p);

    // Copie du résultat sur CPU
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    //MatrixPrint(Mout, n, p);

    cudaMatrixMult<<<grid_size, block_size>>>(d_M, d_M2, d_Mout, n, p);

    cudaMemcpy(M, d_M, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    cudaMemcpy(M2, d_M2, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);

    MatrixPrint(M, n, p);
    MatrixPrint(M2, n, p);
    MatrixPrint(Mout, n, p);

    CPU_time = (double)(end_clock_CPU - start_clock_CPU)/CLOCKS_PER_SEC;

    printf("Temps de calcul CPU:\n");
    printf("%f\n", CPU_time);
    printf("Temps de calcul GPU:\n");
    printf("%1f\n", GPU_time);

    cudaFree(d_M);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    free(M);
    free(M2);
    free(Mout);
}
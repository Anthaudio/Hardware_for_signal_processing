#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>




// Layer 1 
/*
 La fonction suivante initialise une matrice de taille NxP selon différentes options :
    - Si type == 0, initialise la matrice avec des zéros.
    - Si type == 1, initialise la matrice avec des uns.
    - Si type == 2, initialise la matrice comme un kernel prédéfini.
    - Si type == 3, initialise la matrice avec des valeurs aléatoires entre 0 et 1.
 Paramètres :
    - M : pointeur vers la matrice à initialiser.
    - n : nombre de lignes de la matrice.
    - p : nombre de colonnes de la matrice (si n différent de p).
    - d : nombre de kernels de la matrice (profondeur).
    - type : choix de l'initialisation souhaitée.
 */
void MatrixInit(float *M, int n, int p, int d, int type) {
    float random_value;

    if (type == 0) {
        for (int i = 0; i < n * p * d; i++) {
            M[i] = 0;
        }
    }
    else if (type == 1) {
        for (int i = 0; i < n * p * d; i++) {
            M[i] = 1;
        }
    }
    else if (type == 2) {
        for (int i = 0; i < n * p * d; i++) {
            M[i] = 0;
        }
        for (int k = 0; k < d; k++) {
            M[k * (n * p) + (n * p) / 2] = 2;
        }
    }
    else {
        // Valeurs entre 0 et 1
        for (int i = 0; i < n * p * d; i++) {
            random_value = (float)rand() / (float)(RAND_MAX / 1.0);
            M[i] = random_value;
        }
    }
}


void MatrixPrint2D(float *M, int n, int p){
    
    printf("\n");
    for (int lig = 0; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixPrint3D(float *M, int n, int p, int l){
    for(int i =0; i<l; ++i){
        printf("Matrice %d :\n", i+l);
        for(int j =0; j<n; ++j){
            for(int k=0; k<p; ++k){
                printf("%.2f\t", M[i*p*n+k*n+j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


// Layer 2

/*
La fonction cudaConv2D ci-dessous réalise la convolution de la matrice M avec nb_kernel noyaux de convolution, chacun ayant une taille de kernel_size x kernel_size. Les paramètres de la fonction sont les suivants :

    -input: pointeur vers la matrice d'entrée M.
    -kernels: pointeur vers la matrice correspondant aux noyaux de convolution.
    -output: pointeur vers la matrice de sortie Mout.
    -inputwidth: nombre de colonnes de la matrice d'entrée.
    -kernelSize: nombre de lignes et de colonnes du noyau de convolution.
 */


__global__ void cudaConv2D(float *input, float *kernels, float *output, int inputwidth, int kernelSize) {
    int n = gridDim.x;
    int p = gridDim.y;
    // int l = gridDim.z;

    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    float sum = 0.0f;

    for (int ky = 0; ky < kernelSize; ++ky) {
        for (int kx = 0; kx < kernelSize; ++kx) {
            int inputX = x + kx;
            int inputY = y + ky;
            int inputIdx = inputY * inputwidth + inputX;
            int kernelIdx = z * kernelSize * kernelSize + ky * kernelSize + kx;

            sum += input[inputIdx] * kernels[kernelIdx];
        }
    }

    output[outputIdx] = sum;
}


// Layer 3  

/*
  La fonction suivante effectue le MeanPool de la matrice d'entrée M par un kernel 2x2.
  Paramètres :
    - M_ligne : nombre de lignes de la matrice M.
    - M_colonne : nombre de colonnes de la matrice M.
    - M_prof : profondeur de la matrice M.
    - M : rpointeur ves la matrice d'entrée.
    - meanpool_size : nombre de lignes et de colonnes du kernel (noyau de convolution).
    - Mout_ligne : nombre de lignes de la matrice de sortie Mout.
    - Mout_colonne : nombre de colonnes de la matrice de sortie Mout.
    - Mout : pointeur vers la matrice Mout.
  Mout_ligne = M_ligne / meanpool_size.
 */
__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int M_prof, int meanpool_size, int Mout_ligne, int Mout_colonne){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (lig % meanpool_size == 0 && col % meanpool_size == 0){
        
        float s = 0.0;
        int tot_meanpool = meanpool_size * meanpool_size;
        int tot_M = M_ligne * M_colonne;
        int tot_Mout = Mout_ligne * Mout_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            s = 0.0;
            
            for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
                for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot_M] / tot_meanpool;
                }
            }
            
            Mout[(lig / meanpool_size) * Mout_colonne + (col / meanpool_size) + n_prof * tot_Mout] = s;
        }
    }
}


/*
 La fonction suivante applique la fonction d'activation tanh à la matrice M sur le GPU.
 Cette fonction est définie en __device__, elle doit donc être appelée du GPU par une fonction __global__.
 Elle est exécutée sur le GPU.

 Paramètres :
    - M_ligne : nombre de lignes de la matrice M.
    - M_colonne : nombre de colonnes de la matrice M.
    - M_prof : profondeur de la matrice M.
    - M : pointeur vers la matrice.

 Rq: La matrice M est modifiée directement sur le GPU.
 */
__device__ void activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof){
    
    int off_z = M_ligne * M_colonne;

    int index = blockIdx.z * off_z + blockIdx.x * M_colonne + blockIdx.y;

    M[index] = tanh(M[index]);
}


/*
  La fonction suivante appelle la fonction activation_tanh définie juste avant. Elle doit être appelée du GPU.
  Paramètres :
    - M_ligne : nombre de lignes de la matrice M.
    - M_colonne : nombre de colonnes de la matrice M.
    - M_prof : profondeur de la matrice M.
    - M : pointeur vers la matrice.
 */
__global__ void cudaTanh(float* M, int M_ligne, int M_colonne, int M_prof) {
    activation_tanh(M, M_ligne, M_colonne, M_prof);
}




int main(){
    int N = 32;
    int K = 5;
    int D1 = 6;

    int N1 = N - K + 1;
    int N2 = N1 / 2;

    printf("N=%d K=%d D1=%d N1=%d N2=%d", N, K, D1, N1, N2);


    // Dans le CPU
    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(N * N * 1 * sizeof(float));
    
    MatrixInit(raw_data, N, N, 1, 1);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(N1 * N1 * D1 * sizeof(float));
    
    MatrixInit(C1_data, N1, N1, D1, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(N2 * N2 * D1 * sizeof(float));
    
    MatrixInit(S1_data, N2, N2, D1, 0);
    

    float *C1_kernel;    
    C1_kernel = (float*)malloc(K * K * D1 * sizeof(float));
    
    MatrixInit(C1_kernel, K, K, D1, 1);

    
    // Dans le GPU
    
    // Définition des matrices cuda 
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * N * N * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * K * K * D1);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * N1 * N1 * D1);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * N2 * N2 * D1);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * N * N * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * K * K * D1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * N1 * N1 * D1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * N2 * N2 * D1, cudaMemcpyHostToDevice);
  
    dim3 block_size(N1, 1, 1);
    dim3 grid_size(N1, 1, 1);

    dim3 grid_size2(N1, N1, D1);
    
    cudaConv2D<<<grid_size2, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, N, K);
    //
    
    cudaTanh<<<grid_size2, 1>>>(d_C1_data, N1, N1, D1);
    //
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, N1, N1, D1, 2, N2, N2);
    //
    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * N1 * N1 * D1, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * N2 * N2 * D1, cudaMemcpyDeviceToHost);
    
    
    // Affichage de la matrice résultat
    printf("\nMatrice de base raw_data:");
    MatrixPrint2D(raw_data, N, N);
    printf("Noyau de convolution C1_kernel:");
    MatrixPrint2D(C1_kernel, K, K);
    printf("Matrice résultante de la convolution et de la fonction d'activation:");
    MatrixPrint2D(C1_data, N1, N1);
    printf("Matrice résultante du MeanPooling:");
    MatrixPrint2D(S1_data, N2, N2);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}
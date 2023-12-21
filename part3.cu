#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>




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




/*

La fonction suivante sert à effectuer la multiplication matricielle (dot) d'une matrice NxP avec une matrice PxM sur le GPU

Paramètres : 
    n : nombre de lignes de la matrice M1
    p : nombre de colonnes de M1, de lignes de M2
    m : nombre de colonnes de M2
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille PxM,
    Mout : pointeur vers la matrice résultante de la multiplication de taille NxM

On peut considérer les dimensions de la matrice de sortie comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks : n
    les colonnes correspondent aux threads : m
*/
__device__ float* cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if (lig < n && col < m){
        for (int i = 0; i < p; i++){
            s += M1[lig * p + i] * M2[i * m + col];
        }
        Mout[lig * m + col] = s;
    }
    
    return Mout;
}


/*

La fonction suivante sert à additionner deux matrices de même taille NxP sur le GPU 

Paramètres : 
    n : nombre de lignes des matrice,
    p : nombre de colonnes des matrices si n différent de p,
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille NxP,
    Mout : pointeur vers la matrice résultante de l'addition de taille NxP,
    
On peut considérer les dimensions des matrices comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks,
    les colonnes correspondent aux threads
*/
__device__ float* cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < n && col < p){
        Mout[lig * p + col] = M1[lig * p + col] + M2[lig * p + col];
    }
    
    return Mout;
}


__global__ void cudaDense(float* d_M, float* d_Mout, float* d_W, float* d_b, int n, int p, int m){
    
    d_Mout = cudaMatrixMultGeneral(d_M, d_W, d_Mout, n, p, m);
    d_Mout = cudaMatrixAdd(d_Mout, d_b, d_Mout, n, m);
    
}


/*
La fonction cudaFlattenKernel effectue l'aplatissement d'une matrice 2D en un tableau unidimensionnel.
   
Paramètres :
    flattenedArray : pointeur vers le tableau unidimensionnel résultant, alloué sur le GPU,
    matrix : pointeur vers la matrice 2D à aplatir, allouée sur le GPU,
    rows : nombre de lignes de la matrice,
    cols : nombre de colonnes de la matrice.
   
Chaque thread CUDA unique est responsable de copier les éléments d'une matrice 2D dans le tableau unidimensionnel.
La variable tid (thread ID) est calculée en utilisant les indices de bloc (blockIdx.x) et de thread (threadIdx.x).
L'indice du tableau unidimensionnel est mis à jour en conséquence, et les éléments de la matrice sont copiés dans le tableau.
*/
__global__ void cudaFlatten(float* flattenedArray, float** matrix, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Aplatir la matrice en un tableau unidimensionnel
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flattenedArray[tid * (rows * cols) + index++] = matrix[i][j];
        }
    }
}


// Fonction main
int main(){
    
    int N = 32;
    int K = 5;
    int D1 = 6;

    int N1 = N - K + 1;
    int N2 = N1 / 2;
  
    // Dans le CPU
    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 2);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    // Création de la sortie de la conv2D
    float *C2_data;    
    C2_data = (float*)malloc(10 * 10 * 6 * sizeof(float));
    
    MatrixInit(C2_data, 10, 10, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S2_data;    
    S2_data = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 5, 5, 6, 0);
    
    // Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 1);
    
    // Création des poids pour la fin du réseau
    float *W1_kernel;    
    W1_kernel = (float*)malloc(400 * 120 * sizeof(float));
    MatrixInit(W1_kernel, 400, 120, 1, 1);
    
    float *B1_kernel;    
    B1_kernel = (float*)malloc(120 * sizeof(float));
    MatrixInit(B1_kernel, 1, 120, 1, 1);
    
    float *D1_data;    
    D1_data = (float*)malloc(120 * sizeof(float));
    MatrixInit(D1_data, 1, 120, 1, 0);

    
    // Définition des matrices cuda
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data, *d_C2_data, *d_S2_data, *d_D1_data, *d_W1_kernel, *d_B1_kernel;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
    cudaMalloc((void**)&d_C2_data, sizeof(float) * 10 * 10 * 6);
    cudaMalloc((void**)&d_S2_data, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_W1_kernel, sizeof(float) * 400 * 120);
    cudaMalloc((void**)&d_B1_kernel, sizeof(float) * 120);
    cudaMalloc((void**)&d_D1_data, sizeof(float) * 400);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1_kernel, W1_kernel, sizeof(float) * 120 * 400, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1_kernel, B1_kernel, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1_data, D1_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
  

    // Process sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);

    dim3 grid_size2(N1, N1, D1);

    dim3 grid_size_flatten(N2, N2, D1);
    dim3 block_size_flatten(32, 32);
    
    cudaConv2D<<<grid_size2, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 28);
    
    
    cudaTanh<<<grid_size2, 1>>>(d_C1_data, 28, 28, 6);
    
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    
    
    cudaConv2D<<<grid_size2, 1>>>(d_S1_data, d_C1_kernel, d_C2_data, 14, 10);
    
    
    cudaTanh<<<grid_size2, 1>>>(d_C2_data, 10, 10, 16);
    
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C2_data, d_S2_data, 10, 10, 16, 2, 5, 5);
    

    cudaFlatten<<<grid_size_flatten, block_size_flatten>>>(d_S2_data, d_D1_data, 5, 5);


    cudaDense<<<grid_size, block_size>>>(d_C2_data, d_D1_data, d_W1_kernel, d_B1_kernel, 1, 400, 120);


    cudaTanh


    cudaDense(120-->84)


    cudaTanh


    cudaDense(84-->10)


    cudaSoftMax

    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * 10 * 10 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * 5 * 5 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(D1_data, d_D1_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // Affichage de la matrice résultat
    MatrixPrint2D(C2_data, 5, 5);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C2_data);
    cudaFree(d_S2_data);
    cudaFree(d_D1_data);
    cudaFree(d_W1_kernel);
    cudaFree(d_B1_kernel);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(C2_data);
    free(S2_data);
    free(D1_data);
    free(W1_kernel);
    free(B1_kernel);
}
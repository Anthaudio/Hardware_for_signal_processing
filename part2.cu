#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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
 La fonction suivante effectue la convolution de la matrice M avec nb_kernel noyaux de convolution de taille kernel_size x kernel_size.
  Paramètres :
    - M_ligne : nombre de lignes de la matrice M.
    - M_colonne : nombre de colonnes de la matrice M.
    - M : pointeur vers la matrice d'entrée.
    - kernel_size : nombre de lignes et de colonnes du kernel (noyau de convolution).
    - nb_kernel : nombre de kernels (noyaux de convolution).
    - kernel : pointeur vers la matrice correspondant aux kernels.
    - Mout_ligne : nombre de lignes de la matrice de sortie Mout.
    - Mout_colonne : nombre de colonnes de la matrice de sortie Mout.
    - Mout : pointeur vers la matrice de sortie Mout.
 Mout_ligne = (M_ligne - kernel_size) + 1.
 */

/*__global__ void cudaConv2D(float* M, float* kernel, float* Mout,
                        int M_ligne, int M_colonne, int kernel_size,
                        int nb_kernel, int Mout_ligne, int Mout_colonne) {

  // Get thread indices
  int lig = blockIdx.x;
  int col = threadIdx.x;

  // Check if the thread corresponds to a valid output pixel
  if (lig < Mout_ligne && col < Mout_colonne) {
    // Compute the offset of the current output pixel in the output matrix
    int Mout_offset = lig * Mout_colonne + col;

    // Initialize the summation variable
    float sum = 0.0f;

    // Convolve the input matrix and the kernel
    for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
      for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
        for (int n_k = 0; n_k < nb_kernel; n_k++) {
          // Calculate the index of the corresponding input pixel with padding
          int input_row = lig + kernel_lig - kernel_size / 2;
          int input_col = col + kernel_col - kernel_size / 2;

          // Check if the indices are within bounds
          if (input_row >= 0 && input_row < M_ligne && input_col >= 0 && input_col < M_colonne) {
            int input_offset = input_row * M_colonne + input_col + n_k * M_ligne * M_colonne;

            // Calculate the dot product
            float product = M[input_offset] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
            sum += product;
          }
        }
      }
    }

    // Store the convolution result in the output matrix
    Mout[Mout_offset] = sum;
  }
}*/

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
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < M_ligne && col < M_colonne){
        
        int tot_M = M_ligne * M_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            M[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
        }
    }
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
    // Dans le CPU
    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 1);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    

    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 1);

    
    // Dans le GPU
    
    // Définition des matrices cuda 
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
  
    dim3 block_size(28, 1, 1);
    dim3 grid_size(28, 1, 1);

    dim3 grid_size2(28, 28, 6);
    
    cudaConv2D<<<grid_size2, 1>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 5);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28, 28, 6);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Affichage de la matrice résultat
    printf("\nMatrice de base raw_data:");
    MatrixPrint2D(raw_data, 32, 32);
    printf("Noyau de convolution C1_kernel:");
    MatrixPrint2D(C1_kernel, 5, 5);
    printf("Matrice résultante de la convolution et de la fonction d'activation:");
    MatrixPrint3D(C1_data, 28, 28, 6);
    printf("Matrice résultante du MeanPooling:");
    MatrixPrint2D(S1_data, 14, 14);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}
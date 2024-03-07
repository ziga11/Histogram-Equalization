#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define GRAYLEVELS 256
#define err(msg) {fprintf(stderr, "%s\n", msg); exit(1);}
#define INDIR "./Input/"
#define OUTDIR "./OutputParallel/"

#include "stb_image.h"
#include "stb_image_write.h"
    
DIR * d;
struct dirent * dir;

__global__ void findMin(unsigned int * input, unsigned int * output, int n) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared memory with input data, replace zero with UINT_MAX
    sdata[tid] = (i < n) ? (input[i] == 0 ? UINT_MAX : input[i]) : UINT_MAX;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
    
}

/* Thread/Pixel */
__global__ void calculateHistogram(unsigned char * image, unsigned int * histogram, int imageSize) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid < imageSize)
        atomicAdd(&histogram[image[thid]], 1);
}

__global__ void Scan(unsigned int * input, unsigned int * output, int n) {
    /* Balanced trees - Blelloch (1990) - NVIDIA website chapter 39 */
    extern __shared__ unsigned int temp[]; 
    int thid = threadIdx.x; /* ThreadID */
    int offset = 1;

    temp[2 * thid] = (2 * thid < n) ? input[2 * thid] : 0; 
    temp[2 * thid + 1] = (2 * thid + 1 < n) ? input[2 * thid + 1] : 0; 

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads(); 
        if (thid < d) { 
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) { temp[n - 1] = 0; }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            unsigned long t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (2 * thid < n) 
        output[2 * thid] = temp[2 * thid];
    if (2 * thid + 1 < n) 
        output[2 * thid + 1] = temp[2 * thid + 1];
}



__device__ unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize) {
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1));
    
    return (int)scale;
}

__global__ void transformImage(unsigned char * imageIn, unsigned char * imageOut, int width, int height,
                                     unsigned int * cdf, unsigned int cdfMin, unsigned int imageSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        unsigned char pxVal = imageIn[index];
        imageOut[index] = scale(cdf[pxVal], cdfMin, imageSize);
    }
}

unsigned char * fetchImage(int * width, int * height) {
    
    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, "jpg") != NULL || strstr(dir->d_name, "png") != NULL)
            break;
    }
    
    /* When out of files so it doesn't give seg fault*/
    if (dir == NULL) 
        return NULL;

    char * imgName = (char *) calloc(strlen(INDIR) + strlen(dir->d_name) + 1, sizeof(char));
    sprintf(imgName, "%s%s", INDIR, dir->d_name);

    int chCount; /* chCount represents number of channels, since its grayscale we need only 1 --> defined as the last arg in stbi_load*/

    unsigned char * imageIn = stbi_load(imgName, width, height, &chCount, 1);
    
    free(imgName);
    return imageIn;
}

int main() {
    d = opendir(INDIR);

    if (d == NULL)
        err("Failed to open dir");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int width, height, imageSize;
    unsigned char * imageIn;
    
    while ((imageIn = fetchImage(&width, &height)) != NULL) {
        imageSize = width * height;

        cudaEventRecord(start);

        int threadsPerBlock = 128;
        int blocksPerGrid = (imageSize + threadsPerBlock - 1) / threadsPerBlock;
        int numElements = GRAYLEVELS; 

        unsigned int * histogram = (unsigned int *) calloc(numElements, sizeof(unsigned int));
        unsigned int * cdf = (unsigned int *) malloc(numElements * sizeof(unsigned int));
        unsigned char * imageOut = (unsigned char *) malloc(imageSize * sizeof(unsigned char));

        unsigned char * d_imageIn, * d_imageOut;
        unsigned int * d_histogram, * d_cdf;
        unsigned int * d_intermediate, * h_intermediate;
        
        h_intermediate = (unsigned int *) malloc(blocksPerGrid * sizeof(unsigned int));

        cudaMalloc(&d_intermediate, blocksPerGrid * sizeof(unsigned int));
        cudaMalloc(&d_imageIn, imageSize * sizeof(unsigned char));
        cudaMalloc(&d_imageOut, imageSize * sizeof(unsigned char));
        cudaMalloc(&d_histogram, numElements * sizeof(unsigned int));
        cudaMalloc(&d_cdf, numElements * sizeof(unsigned int));
        
        cudaMemset(d_histogram, 0, numElements * sizeof(unsigned int));
        cudaMemset(d_cdf, 0, numElements * sizeof(unsigned int));

        cudaMemcpy(d_imageIn, imageIn, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        calculateHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_imageIn, d_histogram, imageSize);
        cudaMemcpy(histogram, d_histogram, numElements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        Scan<<<1, numElements / 2, numElements * sizeof(unsigned int)>>>(d_histogram, d_cdf, numElements);
        cudaMemcpy(cdf, d_cdf, numElements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        findMin<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(unsigned int)>>>(d_cdf, d_intermediate, numElements);
        cudaMemcpy(h_intermediate, d_intermediate, blocksPerGrid * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        unsigned int minNonZero = UINT_MAX;
        for (int i = 0; i < blocksPerGrid; i++)
            if (h_intermediate[i] < minNonZero)
                minNonZero = h_intermediate[i];


        unsigned int cdfMin = (minNonZero == UINT_MAX) ? 0 : minNonZero;

        dim3 threads(16, 16);
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
        transformImage<<<blocks, threads>>>(d_imageIn, d_imageOut, width, height, d_cdf, cdfMin, imageSize);

        cudaMemcpy(imageOut, d_imageOut, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        char * name = (char *) calloc(1000, sizeof(char));
        sprintf(name, "%s%s", OUTDIR, dir->d_name);
        stbi_write_png(name, width, height, 1, imageOut, width);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Image: %s width:%d height:%d processing time: %f ms\n", dir->d_name, width, height, ms);

        cudaFree(d_imageIn);
        cudaFree(d_imageOut);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        free(name);
        free(imageIn);
        free(imageOut);
        free(histogram);
        free(cdf);
    }
    closedir(d);

    return 0;
}
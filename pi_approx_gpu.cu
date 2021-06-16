#include <stdio.h>
#include <math.h>
#include <iostream>

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__device__ double power(double a, double b){
    return pow(a, b);
}

__global__ void pi_approx_kernel(double n, double *res, double batch_size)
{
    const double index = threadIdx.x + blockDim.x * blockIdx.x;

    // long int threads = blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    const double a = -1;
    const double b = 1;

    const double width = (b-a) / n;

    double x = 0;
    double num = 0;
    double batch_index = batch_size * index;


    for (double i = 0; i < batch_size; i++){
        x = width * (batch_index + i) + a + width/2;
        num += sqrt(1-power(x, 2)) * width;
    }

    atomicAdd(res, num);
}

double pi_approx_cpu(double n){
    double pi = 0;

    double a = -1;
    double b = 1;

    double width = 1/n * (b-a);

    double x;
    double num;

    for (int i = 0; i < n; i++){
        x = (b-a) * i/n + a + width/2;
        num = sqrt(1-pow(x, 2)) * width;

        pi += num;
    }

    pi = 2 * pi;

    return pi;
}

__global__ void pi_approx_gpu_single(double n, double *pi){

    double a = -1;
    double b = 1;

    double width = 1/n * (b-a);

    double x;
    double num;

    for (int i = 0; i < n; i++){
        x = (b-a) * i/n + a + width/2;
        num = sqrt(1-power(x, 2)) * width;

        pi[0] += num;
        
    }
}

double pi_approx_gpu(double iters, int block_count=-1, int thread_count=-1){
    // const double batch_size = iters / (block_count * thread_count);
    double batch_size;

    if (block_count == -1 || thread_count == -1){
        if (iters / (1024 * 1024) >= 1024){
            block_count = 1024;
            thread_count = 1024;
        }
        else {
            block_count = ceil(iters / (1024 * 1024));
            thread_count = iters / block_count / 1024;
        }
    }

    batch_size = ceil(iters / (block_count * thread_count));

    double *d_res = 0;
    double res = 0;

    printf("Batch size: %lf\n\n", batch_size);
    printf("Estimated GPU time in seconds (less accurate for less than 2^30): %lf\n\n", 0.72 * iters / pow(2, 30));
    
    cudaMalloc((void**) &d_res, sizeof(double));
    cudaMemcpy(d_res, &res, sizeof(double), cudaMemcpyHostToDevice);
    

    float elapsed = 0;
    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    pi_approx_kernel<<<block_count, thread_count>>>(iters, d_res, batch_size);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop) );

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("With GPU: %f seconds\n", elapsed / 1000);


    cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    printf("GPU approx: %.50lf\n\n", res * 2);
    
    cudaFree(d_res);

    return res * 2;
}


int main()
{
    double x; 
    std::cout << "2^() iterations: "; // Type a number and press enter
    std::cin >> x; // Get user input from the keyboard

    const double iters = pow(2, x);

    pi_approx_gpu(iters);
    



    // double *d_pi = 0;
    // double pi = 0;

    // cudaMalloc(&d_pi, sizeof(double));


    // HANDLE_ERROR(cudaEventCreate(&start));
    // HANDLE_ERROR(cudaEventCreate(&stop));

    // HANDLE_ERROR(cudaEventRecord(start, 0));

    // pi_approx_gpu_single<<<1, 1>>>(iters, d_pi);

    // cudaMemcpy(&pi, d_pi, sizeof(double), cudaMemcpyDeviceToHost);

    // HANDLE_ERROR(cudaEventRecord(stop, 0));
    // HANDLE_ERROR(cudaEventSynchronize(stop));

    // HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop) );

    // HANDLE_ERROR(cudaEventDestroy(start));
    // HANDLE_ERROR(cudaEventDestroy(stop));

    // printf("With GPU 2: %f seconds\n", elapsed / 1000);

    // printf("GPU approx 2: %.50lf\n\n", 2 * pi);




    clock_t cpu_startTime, cpu_endTime;

    double cpu_ElapseTime=0;
    cpu_startTime = clock();

    double cpu_approx = pi_approx_cpu(iters);

    cpu_endTime = clock();

    cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);

    printf("With CPU: %f seconds\n", cpu_ElapseTime);
    printf("CPU approx: %.50lf\n", cpu_approx);

    
    


    return 0;
}
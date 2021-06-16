# PiApproxCUDA
A calculation of PI using integrals which runs on the GPU (CUDA).

## Compatabilty:
NVIDIA GPUs with compute capabilty of 6.0 or higher.

## Prebuilts:
For GPUs with compute capabilty 8.6 (arch=sm_86) you can use the prebuilts.

## Build it yourself:
To build it yourself, use the following command: `nvcc pi_approx_gpu.cu -arch=sm_{your compute capability here} -o pi_approx_gpu`, for example, if your GPU has a 7.5 compute capability, run: `nvcc pi_approx_gpu.cu -arch=sm_75 -o pi_approx_gpu`.

## Usage:
After compiling / using the prebuilts, just run the `.exe` file (`pi_approx_gpu.exe`) and enter the amount of iterations you want to perform (higher means higher accuracy, and if you entered `30`, it will run `2^30` iterations).

## So, how does this work?
The code calculates the integral of `sqrt(1-x^2)` in the range `-1, 1` which gives you `pi/2`, by calculating the size of many rectangles bellow this function, and the "iterations" you enter are acctualy the number of recantgles to calculate.
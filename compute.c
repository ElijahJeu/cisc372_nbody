#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>


// GPU kernels


__global__ void pairwise_accel_kernel(vector3* pos, double* mass, vector3* accels, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n)
    {
        int index = i * n + j;

        if (i == j)
        {
            accels[index][0] = 0;
            accels[index][1] = 0;
            accels[index][2] = 0;
        }
        else
        {
            vector3 distance;
            distance[0] = pos[i][0] - pos[j][0];
            distance[1] = pos[i][1] - pos[j][1];
            distance[2] = pos[i][2] - pos[j][2];

            double mag_sq = distance[0]*distance[0] +
                            distance[1]*distance[1] +
                            distance[2]*distance[2];

            double mag = sqrt(mag_sq);
            double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / mag_sq;

            accels[index][0] = accelmag * distance[0] / mag;
            accels[index][1] = accelmag * distance[1] / mag;
            accels[index][2] = accelmag * distance[2] / mag;
        }
    }
}

__global__ void update_kernel(vector3* pos, vector3* vel, vector3* accels, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        vector3 accel_sum = {0,0,0};

        for (int j = 0; j < n; j++)
        {
            accel_sum[0] += accels[i*n + j][0];
            accel_sum[1] += accels[i*n + j][1];
            accel_sum[2] += accels[i*n + j][2];
        }

        for (int k = 0; k < 3; k++)
        {
            vel[i][k] += accel_sum[k] * INTERVAL;
            pos[i][k] += vel[i][k] * INTERVAL;
        }
    }
}

// DEVICE POINTERS (global)
static vector3 *d_pos = NULL;
static vector3 *d_vel = NULL;
static vector3 *d_accels = NULL;
static double  *d_mass = NULL;

// REQUIRED INTERFACE
extern "C" void compute()
{
    size_t size_pos = sizeof(vector3) * NUMENTITIES;
    size_t size_mass = sizeof(double) * NUMENTITIES;
    size_t size_matrix = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    // Allocate once (simple version for assignment)
    cudaMalloc((void**)&d_pos, size_pos);
    cudaMalloc((void**)&d_vel, size_pos);
    cudaMalloc((void**)&d_mass, size_mass);
    cudaMalloc((void**)&d_accels, size_matrix);

    // Copy host to  device
    cudaMemcpy(d_pos, hPos, size_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, hVel, size_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, size_mass, cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 block(16,16);
    dim3 grid((NUMENTITIES+15)/16, (NUMENTITIES+15)/16);

    pairwise_accel_kernel<<<grid, block>>>(d_pos, d_mass, d_accels, NUMENTITIES);

    int block1d = 256;
    int grid1d = (NUMENTITIES + block1d - 1) / block1d;

    update_kernel<<<grid1d, block1d>>>(d_pos, d_vel, d_accels, NUMENTITIES);

    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(hPos, d_pos, size_pos, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_vel, size_pos, cudaMemcpyDeviceToHost);

    // Free GPU memory 
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
    cudaFree(d_accels);
}

#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void pairwise_accel_kernel(vector3* pos, double* mass, vector3* accels, int n) {
    // 1. Determine which 'i' and 'j' this specific thread is responsible for
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Make sure we aren't going out of bounds of our array
    if (i < n && j < n) {
        int index = i * n + j; // Flat indexing for our matrix

        if (i == j) {
            accels[index][0] = 0;
            accels[index][1] = 0;
            accels[index][2] = 0;
        } else {
            vector3 distance;
            distance[0] = pos[i][0] - pos[j][0];
            distance[1] = pos[i][1] - pos[j][1];
            distance[2] = pos[i][2] - pos[j][2];

            double mag_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
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
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(vector3* d_pos, vector3* d_vel, double* d_mass, vector3* d_accels)
{
    dim3 block(16,16);
    dim3 grid((NUMENTITIES+15)/16, (NUMENTITIES+15)/16);

    pairwise_accel_kernel<<<grid, block>>>(d_pos, d_mass, d_accels, NUMENTITIES);

    int block1 = 256;
    int grid1 = (NUMENTITIES + block1 - 1) / block1;

    update_kernel<<<grid1, block1>>>(d_pos, d_vel, d_accels, NUMENTITIES);

    cudaDeviceSynchronize();
}
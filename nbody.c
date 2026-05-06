#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// Host memory pointers
vector3 *hVel;
vector3 *hPos;
double *mass;

// Global GPU pointers
// These will be used by compute.cu as 'extern' variables
vector3 *d_hPos, *d_hVel, *d_accels;
double *d_mass;

// initHostMemory: Create storage for entities on the CPU
void initHostMemory(int numObjects)
{
    hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
    hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
    mass = (double *)malloc(sizeof(double) * numObjects);
}

// freeHostMemory: Free storage allocated on the CPU
void freeHostMemory()
{
    free(hVel);
    free(hPos);
    free(mass);
}

// planetFill: Fill with Solar System data
void planetFill(){
    int i, j;
    double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
    for (i=0; i<=NUMPLANETS; i++){
        for (j=0; j<3; j++){
            hPos[i][j] = data[i][j];
            hVel[i][j] = data[i][j+3];
        }
        mass[i] = data[i][6];
    }
}

// randomFill: Fill asteroids randomly
void randomFill(int start, int count)
{
    int i, j;
    for (i = start; i < start + count; i++)
    {
        for (j = 0; j < 3; j++)
        {
            hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
            hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
            mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
        }
    }
}

// printSystem: Prints out the entire system
void printSystem(FILE* handle){
    int i, j;
    for (i=0; i<NUMENTITIES; i++){
        fprintf(handle,"pos=(");
        for (j=0; j<3; j++){
            fprintf(handle,"%lf,", hPos[i][j]);
        }
        fprintf(handle,"),v=(");
        for (j=0; j<3; j++){
            fprintf(handle,"%lf,", hVel[i][j]);
        }
        fprintf(handle,"),m=%lf\n", mass[i]);
    }
    fflush(handle);
}

int main(int argc, char **argv)
{
    clock_t t0 = clock();
    int t_now;
    srand(1234);

    initHostMemory(NUMENTITIES);
    planetFill();
    randomFill(NUMPLANETS + 1, NUMASTEROIDS);

    // 1. Setup GPU memory sizes
    size_t size_pos = sizeof(vector3) * NUMENTITIES;
    size_t size_mass = sizeof(double) * NUMENTITIES;
    size_t size_matrix = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    // 2. Allocate GPU memory
    cudaMalloc((void**)&d_hPos, size_pos);
    cudaMalloc((void**)&d_hVel, size_pos);
    cudaMalloc((void**)&d_mass, size_mass);
    cudaMalloc((void**)&d_accels, size_matrix);

    // 3. Initial Copy from CPU to GPU
    cudaMemcpy(d_hPos, hPos, size_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, size_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, size_mass, cudaMemcpyHostToDevice);

    printf("Starting simulation with %d entities...\n", NUMENTITIES);
    fflush(stdout);

    #ifdef DEBUG
    // Print initial state (CPU version)
    // printSystem(stdout); 
    #endif

    // 4. Main Simulation Loop
    for (t_now = 0; t_now < DURATION; t_now += INTERVAL){
        compute(); // This now just launches kernels in compute.cu
    }

    // 5. Copy final results back to CPU
    cudaMemcpy(hPos, d_hPos, size_pos, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, size_pos, cudaMemcpyDeviceToHost);

    printf("Simulation Finished.\n");
    
    #ifdef DEBUG
    printSystem(stdout);
    #endif

    clock_t t1 = clock() - t0;
    printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

    // 6. Cleanup
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);
    freeHostMemory();

    return 0;
}

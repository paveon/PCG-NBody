/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xpavel34
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <sstream>

#include "nbody.h"
#include "h5Helper.h"
#include "wrappers.cuh"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Time measurement
    struct timeval t1{}, t2{};

    if (argc != 10) {
        printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
        exit(1);
    }

    // Number of particles
    const int N = std::stoi(argv[1]);
    // Length of time step
    const float dt = std::stof(argv[2]);
    // Number of steps
    const size_t steps = std::stoi(argv[3]);
    // Number of thread blocks
    const int thr_blc = std::stoi(argv[4]);
    // Write frequency
    int writeFreq = std::stoi(argv[5]);
    // number of reduction threads
    const int red_thr = std::stoi(argv[6]);
    // Number of reduction threads/blocks
    const int red_thr_blc = std::stoi(argv[7]);

    // Size of the simulation CUDA gird - number of blocks
    const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
    // Size of the reduction CUDA grid - number of blocks
    const size_t reductionGrid = (red_thr + red_thr_blc - 1) / red_thr_blc;

    // Log benchmark setup
    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %zu\n", steps);
    printf("threads/block: %d\n", thr_blc);
    printf("blocks/grid: %lu\n", simulationGrid);
    printf("reduction threads/block: %d\n", red_thr_blc);
    printf("reduction blocks/grid: %lu\n", reductionGrid);

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;

    size_t particleCountRounded = roundUp(N, 32);
    size_t memberArrayByteSize = particleCountRounded * sizeof(float);
    size_t bytesTotal = memberArrayByteSize * t_particles_member_count;
    CudaHostMemoryPool<float> particlesHostPool(bytesTotal, cudaHostAllocWriteCombined);
    t_particles particles_cpu{
            .positionsX = particlesHostPool.data(),
            .positionsY = &particlesHostPool.data()[particleCountRounded],
            .positionsZ = &particlesHostPool.data()[particleCountRounded * 2],
            .velocitiesX = &particlesHostPool.data()[particleCountRounded * 3],
            .velocitiesY = &particlesHostPool.data()[particleCountRounded * 4],
            .velocitiesZ = &particlesHostPool.data()[particleCountRounded * 5],
            .weights = &particlesHostPool.data()[particleCountRounded * 6]
    };

    MemDesc md(
            particles_cpu.positionsX, 1, 0,              // Postition in X
            particles_cpu.positionsY, 1, 0,              // Postition in Y
            particles_cpu.positionsZ, 1, 0,              // Postition in Z
            particles_cpu.velocitiesX, 1, 0,              // Velocity in X
            particles_cpu.velocitiesY, 1, 0,              // Velocity in Y
            particles_cpu.velocitiesZ, 1, 0,              // Velocity in Z
            particles_cpu.weights, 1, 0,              // Weight
            N,                                                                  // Number of particles
            recordsNum);                                                        // Number of records in output file

    // Initialisation of helper class and loading of input data
    auto outputFile = std::string(argv[9]);
    H5Helper h5Helper(argv[8], outputFile, md);

    try {
        h5Helper.init();
        h5Helper.readParticleData();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    memberArrayByteSize = particleCountRounded * sizeof(float);
    bytesTotal = memberArrayByteSize * t_particles_member_count;
    std::vector<CudaDeviceMemoryPool<float>> particleDevicePools;
    particleDevicePools.emplace_back(bytesTotal);
    particleDevicePools.emplace_back(bytesTotal);
    std::vector<t_particles> particles_gpu(2);
    for (auto i = 0; i < particleDevicePools.size(); i++) {
        particles_gpu[i] = {
                .positionsX = particleDevicePools[i].data(),
                .positionsY = &particleDevicePools[i].data()[particleCountRounded],
                .positionsZ = &particleDevicePools[i].data()[particleCountRounded * 2],
                .velocitiesX = &particleDevicePools[i].data()[particleCountRounded * 3],
                .velocitiesY = &particleDevicePools[i].data()[particleCountRounded * 4],
                .velocitiesZ = &particleDevicePools[i].data()[particleCountRounded * 5],
                .weights = &particleDevicePools[i].data()[particleCountRounded * 6]
        };
    }

    cudaMemcpy(particleDevicePools[0].data(), particlesHostPool.data(), particlesHostPool.byteSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(particleDevicePools[1].data(), particlesHostPool.data(), particlesHostPool.byteSize,
               cudaMemcpyHostToDevice);

    gettimeofday(&t1, 0);

    dim3 blockSize(thr_blc);
    dim3 gridSize(simulationGrid);
    for (size_t s = 0; s < steps; s++) {
        calculate_velocity<<<gridSize, blockSize>>>(particles_gpu[s & 1ul], particles_gpu[(s + 1) & 1ul], N, dt);

        if (writeFreq > 0 && (s % writeFreq == 0)) {
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    dim3 gridDimension(ceil(float(inputLength) / float(blockDim.x)));
//    centerOfMass<<<gridDimension, blockDimension>>>(particles_gpu, &comOnGPU.x, &comOnGPU.y, &comOnGPU.z, nullptr, nullptr, 0);


    cudaDeviceSynchronize();

    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnGPU{};
    cudaMemcpy(particlesHostPool.data(), particleDevicePools[steps & 1ul].data(), particlesHostPool.byteSize,
               cudaMemcpyDeviceToHost);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnCPU = centerOfMassCPU(md);

    std::cout << "Center of mass on CPU:" << std::endl
              << comOnCPU.x << ", "
              << comOnCPU.y << ", "
              << comOnCPU.z << ", "
              << comOnCPU.w
              << std::endl;

    std::cout << "Center of mass on GPU:" << std::endl
              << comOnGPU.x << ", "
              << comOnGPU.y << ", "
              << comOnGPU.z << ", "
              << comOnGPU.w
              << std::endl;

    // Writing final values to the file
    h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
    h5Helper.writeParticleDataFinal();

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

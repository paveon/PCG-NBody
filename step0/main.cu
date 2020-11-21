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
    const int steps = std::stoi(argv[3]);
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
    printf("steps: %d\n", steps);
    printf("threads/block: %d\n", thr_blc);
    printf("blocks/grid: %lu\n", simulationGrid);
    printf("reduction threads/block: %d\n", red_thr_blc);
    printf("reduction blocks/grid: %lu\n", reductionGrid);

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                            FILL IN: CPU side memory allocation (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

//    memberArrayByteSize = particleCountRounded * sizeof(float4);
//    bytesTotal = memberArrayByteSize * t_particles_alt_member_count;
//    CudaHostMemoryPool<float4> particlesHostPoolAlt(bytesTotal, cudaHostAllocWriteCombined);
//    t_particles_alt particles_cpu_alt{
//            .positions = particlesHostPoolAlt.data(),
//            .velocities = &particlesHostPoolAlt.data()[particleCountRounded]
//    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                              FILL IN: memory layout descriptor (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     * Caution! Create only after CPU side allocation
     * parameters:
     *                      Stride of two               Offset of the first
     *  Data pointer        consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
    */
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

//    MemDesc md_alt(
//            &particles_cpu_alt.positions[0].x, 4, 0,              // Postition in X
//            &particles_cpu_alt.positions[0].y, 4, 0,              // Postition in Y
//            &particles_cpu_alt.positions[0].z, 4, 0,              // Postition in Z
//            &particles_cpu_alt.velocities[0].x, 4, 0,              // Velocity in X
//            &particles_cpu_alt.velocities[0].y, 4, 0,              // Velocity in Y
//            &particles_cpu_alt.velocities[0].z, 4, 0,              // Velocity in Z
//            &particles_cpu_alt.positions[0].w, 4, 0,              // Weight
//            N,                                                                  // Number of particles
//            recordsNum);                                                        // Number of records in output file

    // Initialisation of helper class and loading of input data
    auto outputFile = std::string(argv[9]);
    H5Helper h5Helper(argv[8], outputFile, md);

//    auto outputFile_alt = outputFile.substr(0, outputFile.length() - 3) + "_alt.h5";
//    H5Helper h5Helper_alt(argv[8], outputFile_alt, md_alt);

    try {
        h5Helper.init();
        h5Helper.readParticleData();

//        h5Helper_alt.init();
//        h5Helper_alt.readParticleData();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                  FILL IN: GPU side memory allocation (step 0)                                    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    memberArrayByteSize = particleCountRounded * sizeof(float);
    bytesTotal = memberArrayByteSize * t_particles_member_count;
    CudaDeviceMemoryPool<float> particlesDevicePool(bytesTotal);
    t_particles particles_gpu{
            .positionsX = particlesDevicePool.data(),
            .positionsY = &particlesDevicePool.data()[particleCountRounded],
            .positionsZ = &particlesDevicePool.data()[particleCountRounded * 2],
            .velocitiesX = &particlesDevicePool.data()[particleCountRounded * 3],
            .velocitiesY = &particlesDevicePool.data()[particleCountRounded * 4],
            .velocitiesZ = &particlesDevicePool.data()[particleCountRounded * 5],
            .weights = &particlesDevicePool.data()[particleCountRounded * 6]
    };

    bytesTotal = memberArrayByteSize * t_velocities_member_count;
    CudaDeviceMemoryPool<float> velocitiesDevicePool(bytesTotal);
    t_velocities velocities_gpu{
            .directionX = velocitiesDevicePool.data(),
            .directionY = &velocitiesDevicePool.data()[particleCountRounded],
            .directionZ = &velocitiesDevicePool.data()[particleCountRounded * 2]
    };


    /// Alternative layout device data
//    memberArrayByteSize = particleCountRounded * sizeof(float4);
//    bytesTotal = memberArrayByteSize * t_particles_alt_member_count;
//    CudaDeviceMemoryPool<float4> particlesDevicePoolAlt(bytesTotal);
//    t_particles_alt particles_gpu_alt{
//            .positions = particlesDevicePoolAlt.data(),
//            .velocities = &particlesDevicePoolAlt.data()[particleCountRounded]
//    };
//
//    bytesTotal = memberArrayByteSize * t_velocities_alt_member_count;
//    CudaDeviceMemoryPool<float4> velocitiesDevicePoolAlt(bytesTotal);
//    t_velocities_alt velocities_gpu_alt{
//            .directions = velocitiesDevicePoolAlt.data(),
//    };


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(particlesDevicePool.data(), particlesHostPool.data(), particlesHostPool.byteSize, cudaMemcpyHostToDevice);
    velocitiesDevicePool.Memset(0);

//    cudaMemcpy(particlesDevicePoolAlt.data(), particlesHostPoolAlt.data(), particlesHostPoolAlt.byteSize, cudaMemcpyHostToDevice);
//    velocitiesDevicePoolAlt.Clear(0);

    gettimeofday(&t1, 0);

    dim3 blockSize(thr_blc);
    dim3 gridSize(simulationGrid);
    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                     //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        calculate_gravitation_velocity<<<gridSize, blockSize>>>(particles_gpu, velocities_gpu, N, dt);
        calculate_collision_velocity<<<gridSize, blockSize>>>(particles_gpu, velocities_gpu, N, dt);
        update_particle<<<gridSize, blockSize>>>(particles_gpu, velocities_gpu, N, dt);

//        calculate_gravitation_velocity2<<<gridSize, blockSize>>>(particles_gpu_alt, velocities_gpu_alt, N, dt);
//        calculate_collision_velocity2<<<gridSize, blockSize>>>(particles_gpu_alt, velocities_gpu_alt, N, dt);
//        update_particle2<<<gridSize, blockSize>>>(particles_gpu_alt, velocities_gpu_alt, N, dt);

        /// Clear device memory for temporary velocities
        velocitiesDevicePool.Memset(0);
//        velocitiesDevicePoolAlt.Clear(0);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                          FILL IN: synchronization  (step 4)                                    //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (writeFreq > 0 && (s % writeFreq == 0)) {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                          FILL IN: synchronization and file access logic (step 4)                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    cudaMemcpy(particlesHostPool.data(), particlesDevicePool.data(), particlesHostPool.byteSize, cudaMemcpyDeviceToHost);
//    cudaMemcpy(particlesHostPoolAlt.data(), particlesDevicePoolAlt.data(), particlesHostPoolAlt.byteSize, cudaMemcpyDeviceToHost);

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

//    h5Helper_alt.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
//    h5Helper_alt.writeParticleDataFinal();

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

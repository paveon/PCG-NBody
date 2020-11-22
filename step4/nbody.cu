/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xpavel34
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate velocity
 * @param p_in       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt) {
    extern __shared__ float cache[];
    float *posX = cache;
    float *posY = &cache[blockDim.x];
    float *posZ = &cache[blockDim.x * 2];
    float *velX = &cache[blockDim.x * 3];
    float *velY = &cache[blockDim.x * 4];
    float *velZ = &cache[blockDim.x * 5];
    float *weights = &cache[blockDim.x * 6];

    unsigned int threadsTotal = gridDim.x * blockDim.x;
    unsigned int gridSteps = ceil(float(N) / threadsTotal);
    unsigned int tileWidth = blockDim.x;
    unsigned int tileCount = ceil(float(N) / tileWidth);

    /// Grid stride loop (if there's not enough total threads to cover all particles)
    for (unsigned int gridIdx = 0; gridIdx < gridSteps; gridIdx++) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        unsigned int globalIdx = (gridIdx * threadsTotal) + (blockIdx.x * blockDim.x + threadIdx.x);

        bool inBounds = globalIdx < N;
        float p1_x = inBounds ? p_in.positionsX[globalIdx] : 0.0f;
        float p1_y = inBounds ? p_in.positionsY[globalIdx] : 0.0f;
        float p1_z = inBounds ? p_in.positionsZ[globalIdx] : 0.0f;
        float p1_vel_x = inBounds ? p_in.velocitiesX[globalIdx] : 0.0f;
        float p1_vel_y = inBounds ? p_in.velocitiesY[globalIdx] : 0.0f;
        float p1_vel_z = inBounds ? p_in.velocitiesZ[globalIdx] : 0.0f;
        float p1_weight = inBounds ? p_in.weights[globalIdx] : 0.0f;

        /// Loop over all tiles with each thread
        for (unsigned int tileIdx = 0; tileIdx < tileCount; tileIdx++) {
            unsigned int tileOffset = tileIdx * blockDim.x;
            unsigned int threadOffset = tileOffset + threadIdx.x;
            posX[threadIdx.x] = (threadOffset < N) ? p_in.positionsX[threadOffset] : 0.0f;
            posY[threadIdx.x] = (threadOffset < N) ? p_in.positionsY[threadOffset] : 0.0f;
            posZ[threadIdx.x] = (threadOffset < N) ? p_in.positionsZ[threadOffset] : 0.0f;
            velX[threadIdx.x] = (threadOffset < N) ? p_in.velocitiesX[threadOffset] : 0.0f;
            velY[threadIdx.x] = (threadOffset < N) ? p_in.velocitiesY[threadOffset] : 0.0f;
            velZ[threadIdx.x] = (threadOffset < N) ? p_in.velocitiesZ[threadOffset] : 0.0f;
            weights[threadIdx.x] = (threadOffset < N) ? p_in.weights[threadOffset] : 0.0f;

            /// Synchronize threads before using shared memory
            __syncthreads();

            /// Loop over all points in a single tile
            for (int p2_idx = 0; p2_idx < tileWidth; p2_idx++) {
//                if (tileOffset + p2_idx >= N) break;
                dx = p1_x - posX[p2_idx];
                dy = p1_y - posY[p2_idx];
                dz = p1_z - posZ[p2_idx];
                float rr = dx * dx + dy * dy + dz * dz;
                float r = sqrt(rr);

                if (r > COLLISION_DISTANCE) {
                    // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
                    // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
                    float r3 = rr * r + FLT_MIN;
                    float G_dt_r3 = -G * dt / r3;
                    float Fg_dt_m2_r = G_dt_r3 * weights[p2_idx];
                    accVelocityX += Fg_dt_m2_r * dx;
                    accVelocityY += Fg_dt_m2_r * dy;
                    accVelocityZ += Fg_dt_m2_r * dz;
                } else {
                    float weightSum = p1_weight + weights[p2_idx];
                    float weightDiff = p1_weight - weights[p2_idx];
                    float p2_w2 = 2 * weights[p2_idx];

                    bool colliding = r > 0.0f;
                    accVelocityX += colliding ? ((p1_vel_x * weightDiff + p2_w2 * velX[p2_idx]) / weightSum) - p1_vel_x
                                              : 0.0f;
                    accVelocityY += colliding ? ((p1_vel_y * weightDiff + p2_w2 * velY[p2_idx]) / weightSum) - p1_vel_y
                                              : 0.0f;
                    accVelocityZ += colliding ? ((p1_vel_z * weightDiff + p2_w2 * velZ[p2_idx]) / weightSum) - p1_vel_z
                                              : 0.0f;
                }
            }

            /// Wait for all threads to finish to avoid overwritten shared memory
            __syncthreads();
        }

        if (globalIdx < N) {
            p_out.velocitiesX[globalIdx] = p1_vel_x + accVelocityX;
            p_out.velocitiesY[globalIdx] = p1_vel_y + accVelocityY;
            p_out.velocitiesZ[globalIdx] = p1_vel_z + accVelocityZ;

            p_out.positionsX[globalIdx] = p1_x + p_out.velocitiesX[globalIdx] * dt;
            p_out.positionsY[globalIdx] = p1_y + p_out.velocitiesY[globalIdx] * dt;
            p_out.positionsZ[globalIdx] = p1_z + p_out.velocitiesZ[globalIdx] * dt;
        }
    }
}// end of calculate_gravitation_velocity


/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void
centerOfMass(t_particles p, float *comX, float *comY, float *comZ, float *comW, int *lock, const int N) {
    unsigned int warpCount = blockDim.x >> 5u;
    extern __shared__ volatile float sharedCOMs[];
    volatile float *sharedX = sharedCOMs;
    volatile float *sharedY = &sharedCOMs[warpCount];
    volatile float *sharedZ = &sharedCOMs[warpCount * 2];
    volatile float *sharedWeights = &sharedCOMs[warpCount * 3];

    unsigned int tx = threadIdx.x;
    unsigned int threadsTotal = gridDim.x * blockDim.x;
    unsigned int gridSteps = ceil(float(N) / threadsTotal);

    /// Local accumulator for COM (in case one thread block runs more than once)
    float4 comLocal = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int gridIdx = 0; gridIdx < gridSteps; gridIdx++) {
        unsigned int globalIdx = (gridIdx * threadsTotal) + (blockIdx.x * blockDim.x + tx);
        bool inBounds = globalIdx < N;
        float weight = inBounds ? p.weights[globalIdx] : 0.0f;
        float dw = (weight > 0.0f) ? 1.0f : 0.0f;
        float posX = inBounds ? (p.positionsX[globalIdx] * dw) : 0.0f;
        float posY = inBounds ? (p.positionsY[globalIdx] * dw) : 0.0f;
        float posZ = inBounds ? (p.positionsZ[globalIdx] * dw) : 0.0f;

        #pragma unroll
        for (unsigned int stride = 16; stride > 0; stride >>= 1u) {
            float w2 = __shfl_down_sync(0xffffffff, weight, stride);
            weight += w2;
            dw = ((weight) > 0.0f) ? (w2 / weight) : 0.0f;
            posX += (__shfl_down_sync(0xffffffff, posX, stride) - posX) * dw;
            posY += (__shfl_down_sync(0xffffffff, posY, stride) - posY) * dw;
            posZ += (__shfl_down_sync(0xffffffff, posZ, stride) - posZ) * dw;
        }

        /// First thread of each warp stores the reduced result to the local variable
        /// to avoid shared memory access when 'gridSteps > 1'
        if ((tx % 32u == 0) && globalIdx < N) {
            dw = ((weight + comLocal.w) > 0.0f) ? (weight / (weight + comLocal.w)) : 0.0f;
            comLocal.x += (posX - comLocal.x) * dw;
            comLocal.y += (posY - comLocal.y) * dw;
            comLocal.z += (posZ - comLocal.z) * dw;
            comLocal.w += weight;
        }
    }

    /// First thread from each warp stores the warp-reduced data into the shared memory
    /// Will cause some bank-conflicts if the block is larger than 1024 threads
    unsigned int warpID = tx >> 5u;
    if (tx % 32u == 0) {
        sharedX[warpID] = comLocal.x;
        sharedY[warpID] = comLocal.y;
        sharedZ[warpID] = comLocal.z;
        sharedWeights[warpID] = comLocal.w;
    }
    __syncthreads();

    /// Block level reduction in shared memory
    for (unsigned int stride = warpCount >> 1ul; stride > 16; stride >>= 1ul) {
        if (tx < stride) {
            float weight = sharedWeights[tx + stride];
            float dw = ((weight + sharedWeights[tx]) > 0.0f) ? (weight / (weight + sharedWeights[tx])) : 0.0f;

            sharedX[tx] += (sharedX[tx + stride] - sharedX[tx]) * dw;
            sharedY[tx] += (sharedY[tx + stride] - sharedY[tx]) * dw;
            sharedZ[tx] += (sharedZ[tx + stride] - sharedZ[tx]) * dw;
            sharedWeights[tx] += sharedWeights[tx + stride];
        }
        __syncthreads();
    }

    if (tx < min(32, warpCount)) {
        float posX = sharedX[tx];
        float posY = sharedY[tx];
        float posZ = sharedZ[tx];
        float weight = sharedWeights[tx];

        for (unsigned int stride = min(16, warpCount >> 1ul); stride > 0; stride >>= 1u) {
            float w2 = __shfl_down_sync(0xffffffff, weight, stride);
            weight += w2;
            float dw = ((weight) > 0.0f) ? (w2 / weight) : 0.0f;
            posX += (__shfl_down_sync(0xffffffff, posX, stride) - posX) * dw;
            posY += (__shfl_down_sync(0xffffffff, posY, stride) - posY) * dw;
            posZ += (__shfl_down_sync(0xffffffff, posZ, stride) - posZ) * dw;
        }

        if (tx == 0) {
            while (atomicExch(lock, 1u) != 0u);  /// Lock

            float dw = ((weight + *comW) > 0.0f) ? (weight / (weight + *comW)) : 0.0f;
            *comX += (posX - *comX) * dw;
            *comY += (posY - *comY) * dw;
            *comZ += (posZ - *comZ) * dw;
            *comW += weight;

            atomicExch(lock, 0u); /// Unlock
        }
    }
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc &memDesc) {
    float4 com = {0, 0, 0, 0};

    for (int i = 0; i < memDesc.getDataSize(); i++) {
        // Calculate the vector on the line connecting points and most recent position of center-of-mass
        const float dx = memDesc.getPosX(i) - com.x;
        const float dy = memDesc.getPosY(i) - com.y;
        const float dz = memDesc.getPosZ(i) - com.z;

        // Calculate weight ratio only if at least one particle isn't massless
        const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                         ? (memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

        // Update position and weight of the center-of-mass according to the weight ration and vector
        com.x += dx * dw;
        com.y += dy * dw;
        com.z += dz * dw;
        com.w += memDesc.getWeight(i);
    }
    return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------

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
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p.positionsX[gID];
        float p1_y = p.positionsY[gID];
        float p1_z = p.positionsZ[gID];
        float p1_weight = p.weights[gID];
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            dx = p1_x - p.positionsX[particleIdx];
            dy = p1_y - p.positionsY[particleIdx];
            dz = p1_z - p.positionsZ[particleIdx];
            float rr = dx * dx + dy * dy + dz * dz;
            float r = sqrt(rr);
            float F = -G * p1_weight * p.weights[particleIdx] / (rr + FLT_MIN);
            float dtw = dt / p1_weight;

            bool notColliding = r > COLLISION_DISTANCE;
            accVelocityX += notColliding ? (F * dx / (r + FLT_MIN)) * dtw : 0.0f;
            accVelocityY += notColliding ? (F * dy / (r + FLT_MIN)) * dtw : 0.0f;
            accVelocityZ += notColliding ? (F * dz / (r + FLT_MIN)) * dtw : 0.0f;
        }

        tmp_vel.directionX[gID] += accVelocityX;
        tmp_vel.directionY[gID] += accVelocityY;
        tmp_vel.directionZ[gID] += accVelocityZ;
    }
}// end of calculate_gravitation_velocity

//Version 2: Use 10 operations, math simplification
//    float r3, G_dt_r3, Fg_dt_m2_r;
//
//    r3 = r * r * r + FLT_MIN;
//
//    // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
//    G_dt_r3 = -G * DT / r3;
//    Fg_dt_m2_r = G_dt_r3 * p2.weight;
//
//    // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
//    vx = Fg_dt_m2_r * dx;
//    vy = Fg_dt_m2_r * dy;
//    vz = Fg_dt_m2_r * dz;


__global__ void calculate_gravitation_velocity2(t_particles_alt p, t_velocities_alt tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p.positions[gID].x;
        float p1_y = p.positions[gID].y;
        float p1_z = p.positions[gID].z;
        float p1_weight = p.positions[gID].w;
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            dx = p1_x - p.positions[particleIdx].x;
            dy = p1_y - p.positions[particleIdx].y;
            dz = p1_z - p.positions[particleIdx].z;
            float rr = dx * dx + dy * dy + dz * dz;
            float r = sqrt(rr);
            float F = -G * p1_weight * p.positions[particleIdx].w / (rr + FLT_MIN);

            bool notColliding = r > COLLISION_DISTANCE;
            accVelocityX += notColliding ? (F * dx / (r + FLT_MIN)) * dt / p1_weight : 0.0f;
            accVelocityY += notColliding ? (F * dy / (r + FLT_MIN)) * dt / p1_weight : 0.0f;
            accVelocityZ += notColliding ? (F * dz / (r + FLT_MIN)) * dt / p1_weight : 0.0f;
        }

        tmp_vel.directions[gID].x += accVelocityX;
        tmp_vel.directions[gID].y += accVelocityY;
        tmp_vel.directions[gID].z += accVelocityZ;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        // Use accumulators to avoid unnecessary global memory access between iterations
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p.positionsX[gID];
        float p1_y = p.positionsY[gID];
        float p1_z = p.positionsZ[gID];
        float p1_vel_x = p.velocitiesX[gID];
        float p1_vel_y = p.velocitiesY[gID];
        float p1_vel_z = p.velocitiesZ[gID];
        float p1_weight = p.weights[gID];
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            float p2_vel_x = p.velocitiesX[particleIdx];
            float p2_vel_y = p.velocitiesY[particleIdx];
            float p2_vel_z = p.velocitiesZ[particleIdx];
            float p2_weight = p.weights[particleIdx];

            float dx = p1_x - p.positionsX[particleIdx];
            float dy = p1_y - p.positionsY[particleIdx];
            float dz = p1_z - p.positionsZ[particleIdx];
            float rr = dx*dx + dy*dy + dz*dz;
            float r = sqrtf(rr);

            // Use temp variables to reduce redundant computations
            float weightSum = p1_weight + p2_weight;
            float weightDiff = p1_weight - p2_weight;
            float p2_w2 = 2 * p2_weight;

            // p1_weight * p1_vel_x - p2_weight * p1_vel_x --> p1_vel_x * (p1_weight - p2_weight)
            // --> p1_vel_x * (weightDiff)

            bool colliding = r > 0.0f && r < COLLISION_DISTANCE;
            accVelocityX += colliding ? ((p1_vel_x * weightDiff + p2_w2 * p2_vel_x) / weightSum) - p1_vel_x : 0.0f;
            accVelocityY += colliding ? ((p1_vel_y * weightDiff + p2_w2 * p2_vel_y) / weightSum) - p1_vel_y : 0.0f;
            accVelocityZ += colliding ? ((p1_vel_z * weightDiff + p2_w2 * p2_vel_z) / weightSum) - p1_vel_z : 0.0f;
        }

        tmp_vel.directionX[gID] += accVelocityX;
        tmp_vel.directionY[gID] += accVelocityY;
        tmp_vel.directionZ[gID] += accVelocityZ;
    }
}// end of calculate_collision_velocity

__global__ void calculate_collision_velocity2(t_particles_alt p, t_velocities_alt tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        // Use accumulators to avoid unnecessary global memory access between iterations
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p.positions[gID].x;
        float p1_y = p.positions[gID].y;
        float p1_z = p.positions[gID].z;
        float p1_weight = p.positions[gID].w;
        float p1_vel_x = p.velocities[gID].x;
        float p1_vel_y = p.velocities[gID].y;
        float p1_vel_z = p.velocities[gID].z;
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            float p2_vel_x = p.velocities[particleIdx].x;
            float p2_vel_y = p.velocities[particleIdx].y;
            float p2_vel_z = p.velocities[particleIdx].z;
            float p2_weight = p.positions[particleIdx].w;

            float dx = p1_x - p.positions[particleIdx].x;
            float dy = p1_y - p.positions[particleIdx].y;
            float dz = p1_z - p.positions[particleIdx].z;
            float rr = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(rr);

            // Use temp variables to reduce redundant computations
            float weightSum = p1_weight + p2_weight;
            float weightDiff = p1_weight - p2_weight;
            float p2_w2 = 2 * p2_weight;

            // p1_weight * p1_vel_x - p2_weight * p1_vel_x --> p1_vel_x * (p1_weight - p2_weight)
            // --> p1_vel_x * (weightDiff)

            bool colliding = r > 0.0f && r < COLLISION_DISTANCE;
            accVelocityX += colliding ? ((p1_vel_x * weightDiff + p2_w2 * p2_vel_x) / weightSum) - p1_vel_x : 0.0f;
            accVelocityY += colliding ? ((p1_vel_y * weightDiff + p2_w2 * p2_vel_y) / weightSum) - p1_vel_y : 0.0f;
            accVelocityZ += colliding ? ((p1_vel_z * weightDiff + p2_w2 * p2_vel_z) / weightSum) - p1_vel_z : 0.0f;
        }

        tmp_vel.directions[gID].x += accVelocityX;
        tmp_vel.directions[gID].y += accVelocityY;
        tmp_vel.directions[gID].z += accVelocityZ;
    }
}
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        p.velocitiesX[gID] += tmp_vel.directionX[gID];
        p.positionsX[gID] += p.velocitiesX[gID] * dt;

        p.velocitiesY[gID] += tmp_vel.directionY[gID];
        p.positionsY[gID] += p.velocitiesY[gID] * dt;

        p.velocitiesZ[gID] += tmp_vel.directionZ[gID];
        p.positionsZ[gID] += p.velocitiesZ[gID] * dt;
    }
}// end of update_particle


__global__ void update_particle2(t_particles_alt p, t_velocities_alt tmp_vel, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        float tmpVelocity = tmp_vel.directions[gID].x;
        p.velocities[gID].x += tmpVelocity;
        p.positions[gID].x += tmpVelocity * dt;

        tmpVelocity = tmp_vel.directions[gID].y;
        p.velocities[gID].y += tmpVelocity;
        p.positions[gID].y += tmpVelocity * dt;

        tmpVelocity = tmp_vel.directions[gID].z;
        p.velocities[gID].y += tmpVelocity;
        p.positions[gID].y += tmpVelocity * dt;
    }
}
//----------------------------------------------------------------------------------------------------------------------

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

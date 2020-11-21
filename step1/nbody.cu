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
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p_in.positionsX[gID];
        float p1_y = p_in.positionsY[gID];
        float p1_z = p_in.positionsZ[gID];
        float p1_vel_x = p_in.velocitiesX[gID];
        float p1_vel_y = p_in.velocitiesY[gID];
        float p1_vel_z = p_in.velocitiesZ[gID];
        float p1_weight = p_in.weights[gID];
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            float p2_weight = p_in.weights[particleIdx];

            dx = p1_x - p_in.positionsX[particleIdx];
            dy = p1_y - p_in.positionsY[particleIdx];
            dz = p1_z - p_in.positionsZ[particleIdx];
            float rr = dx * dx + dy * dy + dz * dz;
            float r = sqrt(rr);

            if (r > COLLISION_DISTANCE) {
                // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
                // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
                float r3 = rr * r + FLT_MIN;
                float G_dt_r3 = -G * dt / r3;
                float Fg_dt_m2_r = G_dt_r3 * p2_weight;
                accVelocityX += Fg_dt_m2_r * dx;
                accVelocityY += Fg_dt_m2_r * dy;
                accVelocityZ += Fg_dt_m2_r * dz;
            }
            else {
                float p2_vel_x = p_in.velocitiesX[particleIdx];
                float p2_vel_y = p_in.velocitiesY[particleIdx];
                float p2_vel_z = p_in.velocitiesZ[particleIdx];

                float weightSum = p1_weight + p2_weight;
                float weightDiff = p1_weight - p2_weight;
                float p2_w2 = 2 * p2_weight;

                bool colliding = r > 0.0f;
                accVelocityX += colliding ? ((p1_vel_x * weightDiff + p2_w2 * p2_vel_x) / weightSum) - p1_vel_x : 0.0f;
                accVelocityY += colliding ? ((p1_vel_y * weightDiff + p2_w2 * p2_vel_y) / weightSum) - p1_vel_y : 0.0f;
                accVelocityZ += colliding ? ((p1_vel_z * weightDiff + p2_w2 * p2_vel_z) / weightSum) - p1_vel_z : 0.0f;
            }
        }

        p_out.velocitiesX[gID] = p1_vel_x + accVelocityX;
        p_out.velocitiesY[gID] = p1_vel_y + accVelocityY;
        p_out.velocitiesZ[gID] = p1_vel_z + accVelocityZ;

        p_out.positionsX[gID] = p1_x + p_out.velocitiesX[gID] * dt;
        p_out.positionsY[gID] = p1_y + p_out.velocitiesY[gID] * dt;
        p_out.positionsZ[gID] = p1_z + p_out.velocitiesZ[gID] * dt;
    }
}// end of calculate_gravitation_velocity

__global__ void calculate_velocity2(t_particles p_in, t_particles p_out, int N, float dt) {
    for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        float p1_x = p_in.positionsX[gID];
        float p1_y = p_in.positionsY[gID];
        float p1_z = p_in.positionsZ[gID];
        float p1_vel_x = p_in.velocitiesX[gID];
        float p1_vel_y = p_in.velocitiesY[gID];
        float p1_vel_z = p_in.velocitiesZ[gID];
        float p1_weight = p_in.weights[gID];
        for (int particleIdx = 0; particleIdx < N; particleIdx++) {
            float p2_weight = p_in.weights[particleIdx];
            float p2_vel_x = p_in.velocitiesX[particleIdx];
            float p2_vel_y = p_in.velocitiesY[particleIdx];
            float p2_vel_z = p_in.velocitiesZ[particleIdx];

            dx = p1_x - p_in.positionsX[particleIdx];
            dy = p1_y - p_in.positionsY[particleIdx];
            dz = p1_z - p_in.positionsZ[particleIdx];
            float rr = dx * dx + dy * dy + dz * dz;
            float r = sqrt(rr);
            float r3 = rr * r + FLT_MIN;
            float G_dt_r3 = -G * dt / r3;
            float Fg_dt_m2_r = G_dt_r3 * p2_weight;
            float weightSum = p1_weight + p2_weight;
            float weightDiff = p1_weight - p2_weight;
            float p2_w2 = 2 * p2_weight;

            bool notColliding = r > COLLISION_DISTANCE;
            accVelocityX += notColliding ? Fg_dt_m2_r * dx : ((p1_vel_x * weightDiff + p2_w2 * p2_vel_x) / weightSum) - p1_vel_x;
            accVelocityY += notColliding ? Fg_dt_m2_r * dy : ((p1_vel_y * weightDiff + p2_w2 * p2_vel_y) / weightSum) - p1_vel_y;
            accVelocityZ += notColliding ? Fg_dt_m2_r * dz : ((p1_vel_z * weightDiff + p2_w2 * p2_vel_z) / weightSum) - p1_vel_z;
        }

        p_out.velocitiesX[gID] = p1_vel_x + accVelocityX;
        p_out.velocitiesY[gID] = p1_vel_y + accVelocityY;
        p_out.velocitiesZ[gID] = p1_vel_z + accVelocityZ;

        p_out.positionsX[gID] = p1_x + p_out.velocitiesX[gID] * dt;
        p_out.positionsY[gID] = p1_y + p_out.velocitiesY[gID] * dt;
        p_out.positionsZ[gID] = p1_z + p_out.velocitiesZ[gID] * dt;
    }
}// end of calculate_gravitation_velocity
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

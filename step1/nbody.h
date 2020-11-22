/**
 * @File nbody.h
 *
 * Header file of your implementation to the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xpavel34
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include "h5Helper.h"

/* Gravitation constant */
constexpr float G =  6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;


/**
 * Particles data structure
 */
typedef struct
{
    float* positionsX;
    float* positionsY;
    float* positionsZ;
    float* velocitiesX;
    float* velocitiesY;
    float* velocitiesZ;
    float* weights;
} t_particles;

constexpr size_t t_particles_member_count = 7;


/**
 * Velocities data structure (to be used as buffer for partial results)
 */
typedef struct
{
    float* directionX;
    float* directionY;
    float* directionZ;
} t_velocities;

constexpr size_t t_velocities_member_count = 3;


/**
 * CUDA kernel to calculate velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p_in,
                                   t_particles p_out,
                                   int N,
                                   float dt);

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
__global__ void centerOfMass(t_particles p,
                             float*      comX,
                             float*      comY,
                             float*      comZ,
                             float*      comW,
                             int*        lock,
                             const int   N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */

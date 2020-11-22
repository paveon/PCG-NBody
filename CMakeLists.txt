cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-8")

project(PCG_CUDA LANGUAGES C CXX CUDA)

find_package(HDF5 REQUIRED)

set(TARGETS 0 1 2 3.1 3.2 4)

foreach(target_id IN LISTS TARGETS)
    set(STEP_TARGET_${target_id} nbody_${target_id})
    set(SOURCES commons/h5Helper.cpp step${target_id}/main.cu step${target_id}/nbody.cu)
    set(HEADERS commons/h5Helper.h commons/wrappers.cuh step${target_id}/nbody.h)
    add_executable(${STEP_TARGET_${target_id}} ${SOURCES} ${HEADERS})
    target_link_libraries(${STEP_TARGET_${target_id}} PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(${STEP_TARGET_${target_id}} PUBLIC commons/ ${HDF5_INCLUDE_DIRS})
    set_target_properties(${STEP_TARGET_${target_id}} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()



add_executable(gen commons/gen.cpp commons/h5Helper.cpp)
target_link_libraries(gen PUBLIC ${HDF5_LIBRARIES})
target_include_directories(gen PUBLIC commons ${HDF5_INCLUDE_DIRS})

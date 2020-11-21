cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-8")

project(PCG_CUDA LANGUAGES C CXX CUDA)

#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CXX_STANDARD 11)
#set( CMAKE_VERBOSE_MAKEFILE on )

#add_subdirectory(libwb)
#add_executable(generator dataset_generator.cpp)
#target_link_libraries(generator wb)
#target_include_directories(generator PUBLIC libwb)

#add_executable(test_cuda main.cu)
#target_link_libraries(test_cuda wb)
#target_include_directories(test_cuda PUBLIC libwb)

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

#set(STEP0_TARGET nbody_0)
#set(SOURCES commons/h5Helper.cpp step0/main.cu step0/nbody.cu)
#set(HEADERS commons/h5Helper.h commons/wrappers.cuh step0/nbody.h)
#add_executable(${STEP0_TARGET} ${SOURCES} ${HEADERS})
#target_link_libraries(${STEP0_TARGET} PUBLIC ${HDF5_LIBRARIES})
#target_include_directories(${STEP0_TARGET} PUBLIC commons/ ${HDF5_INCLUDE_DIRS})
#
#set(STEP1_TARGET nbody_1)
#set(SOURCES commons/h5Helper.cpp step1/main.cu step1/nbody.cu)
#set(HEADERS commons/h5Helper.h commons/wrappers.cuh step1/nbody.h)
#add_executable(${STEP1_TARGET} ${SOURCES} ${HEADERS})
#target_link_libraries(${STEP1_TARGET} PUBLIC ${HDF5_LIBRARIES})
#target_include_directories(${STEP1_TARGET} PUBLIC commons/ ${HDF5_INCLUDE_DIRS})
#
#set(STEP2_TARGET nbody_2)
#set(SOURCES commons/h5Helper.cpp step2/main.cu step2/nbody.cu)
#set(HEADERS commons/h5Helper.h commons/wrappers.cuh step2/nbody.h)
#add_executable(${STEP2_TARGET} ${SOURCES} ${HEADERS})
#target_link_libraries(${STEP2_TARGET} PUBLIC ${HDF5_LIBRARIES})
#target_include_directories(${STEP2_TARGET} PUBLIC commons/ ${HDF5_INCLUDE_DIRS})


add_executable(gen commons/gen.cpp commons/h5Helper.cpp)
target_link_libraries(gen PUBLIC ${HDF5_LIBRARIES})
target_include_directories(gen PUBLIC commons ${HDF5_INCLUDE_DIRS})

#set_target_properties(${STEP0_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(${STEP1_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(${STEP2_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
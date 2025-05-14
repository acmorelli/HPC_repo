cmake_minimum_required(VERSION 3.10)
project(AllgatherMerge LANGUAGES C CXX)

# Find the MPI package
find_package(MPI REQUIRED)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release)" FORCE)
endif()

# Set the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# You may append any additional source files here
add_executable(allgather_merge main.cpp baseline.cpp algorithm1.cpp algorithm2.cpp)


# DO NOT MODIFY BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
# Target include directories
target_include_directories(allgather_merge PRIVATE ${MPI_C_INCLUDE_PATH})

# Link the MPI C++ library
target_link_libraries(allgather_merge PRIVATE MPI::MPI_CXX)
target_link_libraries(allgather_merge PRIVATE m)

# Set build-specific compiler options and definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(allgather_merge PRIVATE DEBUG)
    target_compile_options(allgather_merge PRIVATE -g -O0 -Wall -Wextra -pedantic)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(allgather_merge PRIVATE -O3 -march=native -Wall -pedantic)
endif()

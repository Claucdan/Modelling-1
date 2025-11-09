cmake_minimum_required(VERSION 3.10)
project(SchrodingerSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -pedantic)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(schrodinger main.cpp)
target_link_libraries(schrodinger PRIVATE Eigen3::Eigen)


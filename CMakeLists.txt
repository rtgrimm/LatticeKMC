cmake_minimum_required(VERSION 3.21)
project(KMC)

set(CMAKE_CXX_STANDARD 23)

add_executable(Catch_tests_run Native/tests.cpp)
add_executable(KMC Native/main.cpp Native/lattice.hpp)
target_link_libraries(KMC tbb)
target_link_libraries(Catch_tests_run tbb)

find_package(SWIG REQUIRED)
include(UseSWIG)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

set_source_files_properties(Nano.i PROPERTIES
        CPLUSPLUS ON
        INCLUDE_DIRECTORIES ./
        )

swig_add_library(Nano
        TYPE SHARED
        LANGUAGE python
        SOURCES	Nano.i)

include_directories(${SWIG_MODULE_Nano_REAL_NAME}  "./")
target_include_directories(${SWIG_MODULE_Nano_REAL_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} "./")
target_link_libraries(${SWIG_MODULE_Nano_REAL_NAME} PRIVATE ${PYTHON_LIBRARIES} tbb)
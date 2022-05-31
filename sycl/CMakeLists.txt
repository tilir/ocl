#------------------------------------------------------------------------------
#
# CMake build system for DPC++/SYCL examples
#
# on linux (or windows with unix makefiles)
# run as: cmake -DCMAKE_CXX_COMPILER=${SYCL_COMPILER}/clang++ ..
#
# on windows it is like
# with Visual Studio: cmake -T "Intel(R) oneAPI DPC++ Compiler" ..
# with Ninja: cmake -GNinja -DCMAKE_CXX_COMPILER=dpcpp ..
#
# on Windows run tests as: cmake --build . --target test
# also: ctest --force-new-ctest-process --rerun-failed --output-on-failure
#
# Running on Windows with env
# > set "SYCL_DEVICE_FILTER=cpu" & sgemm\matmult_local.exe
# also set "SYCL_CACHE_PERSISTENT=1" is useful to not spend time recompiling
#
#------------------------------------------------------------------------------
#
#  Macros to control things:
#  -DUSE_BOOST=1 if you have installed findable boost (for better option parser)
#  -DRUNHOST=1 run without GPU (SYCL as a host code)
#  -DMEASURE_NORMAL=1 to measure with reference code
#  -DVERIFY=1 to verify results (for instance that sorting actually sorted smth)
#  -DINORD=1 to use inorder queues
#  -DUSE_MKL=1 to build MKL-dependent kernels
#  -DUSE_CIMG=1 to build CIMG-dependent kernels
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

# for release measurements try:
# cmake -GNinja -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DMEASURE_NORMAL=1 -DVERIFY=1 -DUSE_CIMG=1 -DUSE_MKL=1 -DDUMP_SPIRV=1 ..
# cmake -GNinja -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DUSE_CIMG=1 -DUSE_MKL=1 -DDUMP_SPIRV=1 ..

cmake_minimum_required(VERSION 3.15)
project(syclexamples)

if(USE_BOOST)
set(Boost_USE_MULTITHREADED ON)
endif()

if(USE_MKL)
find_package(MKL CONFIG REQUIRED)
endif()

# Ugliest hack possible for stable Windows builds
# Idea is: we do not have regular linker syntax:
# 'dpcpp target.obj /link <linker args>' works, but 'dpcpp target.obj <linker args>' not.
# So we have to redefine CMAKE_CXX_LINK_EXECUTABLE target, which is strongly disregarded of course.
# Madness, freaking madness.
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc")
  set(CMAKE_EXE_LINKER_FLAGS "-fsycl")
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /nologo <DEFINES> <INCLUDES> /EHsc <FLAGS> /Fo<OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib /nologo <OBJECTS> /out:<TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl /nologo <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
else()
  set(CMAKE_EXE_LINKER_FLAGS "-fsycl")
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

if(USE_BOOST)
find_package(Boost COMPONENTS
             program_options
             # other if required
             REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
endif()

# function to build single kernel
# RUNHOST and MEASURE_NORMAL env consumed here
include(cmake/buildv.cmake)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# interface testers library
add_library(testers-frame INTERFACE)
target_include_directories(testers-frame INTERFACE framework)
if(USE_CIMG)
target_include_directories(testers-frame INTERFACE ${ROOT_DIR}/cimg)
endif()

# enable testing shall go before add_subdir
# https://stackoverflow.com/questions/30250494/ctest-not-detecting-tests
enable_testing()

# vector additions
add_subdirectory(vadd)

# bitnoic sorts
add_subdirectory(bitonic)

# matrix multiplications
add_subdirectory(sgemm)

# histograms
add_subdirectory(histogram)

# reductions
add_subdirectory(reduction)

# image filtering and samplers
if(USE_CIMG)
add_subdirectory(filtering)
endif()

# dump folder for shader dumps if any
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps)

# dummy install
install()
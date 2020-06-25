#-------------------------------------------------------------------------------
#
# FrameCL build system -- main entry
#
#-------------------------------------------------------------------------------
#
# Structure of build is:
# root
#    /include    -- for includes
#    /kernels    -- for kernel examples
#
# You shall use CMAKE_BUILD_TYPE = DEBUG | RELEASE when building
#
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

project(framecl)

set(Boost_USE_MULTITHREADED ON)
set(CMAKE_CXX_STANDARD 20)

find_package(Boost COMPONENTS
             program_options
             # other if required
             REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

option(BUILD_CIMG_KERNELS "Build kernel examples using CImg" ON)
option(INCLUDE_INTEL_TESTS "Build kernel examples using CImg" ON)
option(INCLUDE_NVIDIA_TESTS "Build kernel examples using CImg" ON)
option(INCLUDE_FAILING_TESTS "Build kernel examples using CImg" OFF)

# TODO: will we support Windows build? Not yet, I think...

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(BUILD_CIMG_KERNELS)
  find_package(X11 REQUIRED)
  include_directories(${X11_INCLUDE_DIR})

  find_package(JPEG REQUIRED)
  include_directories(${JPEG_INCLUDE_DIR})
endif(BUILD_CIMG_KERNELS)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

# just for cmake debug
if(DUMPVARS)
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endif()

# enable ctest runs
enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

# collecting kernels
add_subdirectory(kernels)
message(STATUS "Collected kernels: ${KERNELS}")
message(STATUS "Collected test names: ${KERNELTESTNAMES}")
message(STATUS "Collected test lines: ${KERNELTESTS}")

# common includes and libs
foreach(KERNEL ${KERNELS})
  target_compile_options(${KERNEL} PUBLIC "-Wall" "-Wextra" "-pedantic" "-Wfatal-errors")
  target_include_directories(${KERNEL} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${KERNEL} Boost::program_options)
  if (BUILD_CIMG_KERNELS)
    target_link_libraries(${KERNEL} ${X11_LIBRARIES})
    target_link_libraries(${KERNEL} ${JPEG_LIBRARY})
  endif()
  target_link_libraries(${KERNEL} Threads::Threads)
  target_link_libraries(${KERNEL} ${OpenCL_LIBRARIES})
endforeach()

# dummy install
install()

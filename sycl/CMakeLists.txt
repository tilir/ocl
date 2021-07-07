# run as cmake -DCMAKE_CXX_COMPILER=${DPC_COMPILER}/clang++ ..
# TODO: now this don't work because cmake wants two-step compilation

cmake_minimum_required(VERSION 3.14)

project(syclexamples)

set(Boost_USE_MULTITHREADED ON)
set(CMAKE_CXX_STANDARD 20)

find_package(Boost COMPONENTS
             program_options
             # other if required
             REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(KERNELS
  vectoradd
  vectoradd_complexdeps
  vectoradd_devicemem
  vectoradd_inord
  vectoradd_sharedmem
  vectoradd_wait
)

# common includes and libs
foreach(KERNEL ${KERNELS})
  add_executable(${KERNEL} ${KERNEL}.cc)
  target_compile_options(${KERNEL} PUBLIC "-fsycl" "-fsycl-unnamed-lambda")
  target_link_libraries(${KERNEL} Boost::program_options)
endforeach()

# dummy install
install()



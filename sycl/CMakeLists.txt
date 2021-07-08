# run as cmake -DCMAKE_CXX_COMPILER=${DPC_COMPILER}/clang++ ..

cmake_minimum_required(VERSION 3.14)

project(syclexamples)

set(Boost_USE_MULTITHREADED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fsycl")

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
  vectoradd_inequal
  vectoradd_simd
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps)

# common includes and libs
foreach(KERNEL ${KERNELS})
  add_executable(${KERNEL} ${KERNEL}.cc)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/${KERNEL})
  target_compile_options(${KERNEL} PUBLIC "-fsycl-unnamed-lambda")
  target_link_libraries(${KERNEL} Boost::program_options)

  add_custom_target(${KERNEL}_run ALL
     COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/${KERNEL}" ${CMAKE_BINARY_DIR}/${KERNEL}
     DEPENDS ${KERNEL} dumps/${KERNEL}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endforeach()

# special executable for host mem allocation
add_executable(vectoradd_hostmem vectoradd_sharedmem.cc)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/vectoradd_hostmem)
target_compile_options(vectoradd_hostmem PUBLIC "-fsycl-unnamed-lambda" "-DHOST_ALLOC")
target_link_libraries(vectoradd_hostmem Boost::program_options)
add_custom_target(vectoradd_hostmem_run ALL
   COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/vectoradd_hostmem" ${CMAKE_BINARY_DIR}/vectoradd_hostmem
   DEPENDS vectoradd_hostmem dumps/vectoradd_hostmem
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# dummy install
install()


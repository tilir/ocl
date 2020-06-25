set(MULT_SRCS
  mmult.cc
)

add_executable(mmult ${MULT_SRCS})
add_dependencies(check mmult)

if(INCLUDE_INTEL_TESTS)
  add_test(NAME "mmult_simple"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/mmult/simple.cl --check)
  add_test(NAME "mmult_local"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/mmult/localmem.cl --check)
endif(INCLUDE_INTEL_TESTS)  

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "mmult_simple_nv"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/mmult/simple.cl --check)
  add_test(NAME "mmult_local_nv"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/mmult/localmem.cl --check)
endif(INCLUDE_NVIDIA_TESTS)

# register kernels for level up
set(KERNELS ${KERNELS} mmult PARENT_SCOPE)

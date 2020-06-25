set(SRCS
  reduce.cc
)

add_executable(reduce ${SRCS})
add_dependencies(check reduce)

if(INCLUDE_INTEL_TESTS)
  add_test(NAME "reduce"
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/reduce --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/reduce/reduce.cl --check)
endif(INCLUDE_INTEL_TESTS)

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "reduce_nv" 
           COMMAND ${CMAKE_CURRENT_BIN_DIR}/reduce --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/reduce/reduce.cl --check)
endif(INCLUDE_NVIDIA_TESTS)

# register kernels for level up
set(KERNELS ${KERNELS} reduce PARENT_SCOPE)

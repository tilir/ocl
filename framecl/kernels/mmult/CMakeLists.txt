set(MULT_SRCS
  mmult.cc
)

add_executable(mmult ${MULT_SRCS})
add_dependencies(check mmult)

add_test(NAME "mmult_simple"
         COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/mmult/simple.cl --check)
add_test(NAME "mmult_local"
         COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/mmult/localmem.cl --check)

add_test(NAME "mmult_simple_nv"
         COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/mmult/simple.cl --check)
add_test(NAME "mmult_local_nv"
         COMMAND ${CMAKE_CURRENT_BIN_DIR}/mmult --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/mmult/localmem.cl --check)

# register kernels for level up
set(KERNELS ${KERNELS} mmult PARENT_SCOPE)

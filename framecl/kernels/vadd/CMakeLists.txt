set(SRCS
  vadd.cc
)

add_executable(vadd ${SRCS})
add_dependencies(check vadd)

if(INCLUDE_INTEL_TESTS)
  add_test(NAME "vadd" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
endif(INCLUDE_INTEL_TESTS)

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "vadd_nv" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
endif(INCLUDE_NVIDIA_TESTS)

# register kernels for level up
set(KERNELS ${KERNELS} vadd PARENT_SCOPE)
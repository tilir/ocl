# build system fragment, see upper CMakeLists for whole picture

add_executable(vadd vadd.cc)
add_executable(vadd_huge vadd_huge.cc)
add_dependencies(check vadd vadd_huge)

if(INCLUDE_INTEL_TESTS)
  add_test(NAME "vadd" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
  add_test(NAME "vadd_huge" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd_huge --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
endif(INCLUDE_INTEL_TESTS)

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "vadd_nv" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
  add_test(NAME "vadd_huge_nv" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd_huge --platform NVIDIA --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)
endif(INCLUDE_NVIDIA_TESTS)

# register kernels for level up
set(KERNELS ${KERNELS} vadd vadd_huge PARENT_SCOPE)

set(SRCS
  justbuf.cc
)

add_executable(justbuf ${SRCS})
add_dependencies(check justbuf)

if(INCLUDE_INTEL_TESTS)
  add_test(NAME "justbuf" COMMAND ${CMAKE_CURRENT_BINARY_DIR}/justbuf --platform Intel --check)
endif(INCLUDE_INTEL_TESTS)  

if(INCLUDE_NVIDIA_TESTS)
  add_test(NAME "justbuf_nv" COMMAND ${CMAKE_CURRENT_BINARY_DIR}/justbuf --platform NVIDIA --check)
endif(INCLUDE_NVIDIA_TESTS)

set(KERNELS ${KERNELS} justbuf PARENT_SCOPE)

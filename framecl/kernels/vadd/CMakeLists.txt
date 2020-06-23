set(SRCS
  vadd.cc
)

add_executable(vadd ${SRCS})
add_dependencies(check vadd)

add_test(NAME "vadd" COMMAND ${CMAKE_CURRENT_BIN_DIR}/vadd --platform Intel --program ${CMAKE_SOURCE_DIR}/kernels/vadd/vadd.cl --check)

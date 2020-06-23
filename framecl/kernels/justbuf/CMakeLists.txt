set(SRCS
  justbuf.cc
)

add_executable(justbuf ${SRCS})
add_dependencies(check justbuf)

add_test(NAME "justbuf" COMMAND ${CMAKE_CURRENT_BINARY_DIR}/justbuf --platform Intel --check)

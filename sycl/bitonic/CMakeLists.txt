set(KERNELS
  bitonicsort
  bitonic_shared
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(bitonic_simple bitonicsort.cc "SIMPLERANGE=1")
buildv(bitonic_host bitonic_shared.cc "HOST_ALLOC=1")

set(TESTING
  bitonicsort
  bitonic_shared
  bitonic_simple
  bitonic_host
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
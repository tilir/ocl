set(KERNELS
  bitonicsort
  bitonic_shared
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING
  bitonicsort
  bitonic_shared
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
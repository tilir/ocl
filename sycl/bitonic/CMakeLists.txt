set(KERNELS
  bitonic_buffer
  bitonic_device
  bitonic_device_local
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING
  bitonic_buffer
  bitonic_device
  bitonic_device_local
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
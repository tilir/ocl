set(KERNELS
  hist_naive
  hist_naive_acc
  hist_local
  hist_private
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING
  hist_naive
  hist_naive_acc
  hist_local
  hist_private
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
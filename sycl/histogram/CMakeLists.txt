set(KERNELS
  hist_naive
  hist_naive_acc
  hist_local
  hist_local_acc
  hist_local_acc_spec
  hist_private
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(hist_naive_host hist_naive_acc.cc "HOST_PTR=1")

set(TESTING
  hist_naive
  hist_naive_acc
  hist_naive_host
  hist_local
  hist_local_acc
  hist_local_acc_spec
  hist_private
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
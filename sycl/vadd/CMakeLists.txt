set(KERNELS
  vectoradd
  vectoradd_complexdeps
  vectoradd_devicemem
  vectoradd_sharedmem
  vectoradd_wait  
  vectoradd_simd
# excluded from testing:
  vectoradd_inequal
)

# build kernels
# add test: just quiet run to be sure all is ok
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special parameters
buildv(vectoradd_hostmem vectoradd_sharedmem.cc "HOST_ALLOC=1")
buildv(vectoradd_inord vectoradd_wait.cc "INORD=1")
# excluded from testing:
buildv(vectoradd_nowait vectoradd_wait.cc "NOWAIT=1")
buildv(vectoradd_inord_nowait vectoradd_wait.cc "NOWAIT=1" "INORD=1")

set(TESTING
  vectoradd
  vectoradd_complexdeps
  vectoradd_devicemem
  vectoradd_sharedmem
  vectoradd_hostmem
  vectoradd_inord
  vectoradd_wait 
  vectoradd_simd
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -quiet)
endforeach()
set(KERNELS
  boolmachines
)

# build kernels
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

set(TESTING  
  boolmachines
)

foreach(KERNEL ${TESTING})
  add_test(NAME ${KERNEL}_run
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} -randboxes=512 -randfilter=3 -novis -quiet)
endforeach()
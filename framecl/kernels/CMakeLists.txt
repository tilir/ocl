set(KERNELS
  justbuf
  vadd
  hist
)

foreach(KERNEL ${KERNELS})
  add_subdirectory(${KERNEL})
endforeach()

set(KERNELS "${KERNELS}" PARENT_SCOPE)


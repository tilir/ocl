set(SUBDIRS
  justbuf
  vadd
  reduce
  hist
  mmult
  jigsaw
)

set(KERNELS)

foreach(SUBDIR ${SUBDIRS})
  add_subdirectory(${SUBDIR})
endforeach()

set(KERNELS "${KERNELS}" PARENT_SCOPE)


set(SUBDIRS
  justbuf
  vadd
  hist
)

set(KERNELS)

foreach(SUBDIR ${SUBDIRS})
  add_subdirectory(${SUBDIR})
endforeach()

set(KERNELS "${KERNELS}" PARENT_SCOPE)


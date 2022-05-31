#------------------------------------------------------------------------------
#
# CMake build system for DPC++/SYCL examples
#
# builds function: adds compiler definitions
# expect 1-3 arguments: target [define] [define]
#
#------------------------------------------------------------------------------

function(builds KERNEL)
if (NOT "${ARGV1}" STREQUAL "")
  target_compile_definitions(${KERNEL} PRIVATE ${ARGV1})
endif()
if (NOT "${ARGV2}" STREQUAL "")
  target_compile_definitions(${KERNEL} PRIVATE ${ARGV2})
endif()
if (WIN32)
  # for CImg on Windows, suppressing some warnings
  target_compile_definitions(${KERNEL} PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${KERNEL} PUBLIC "-Wno-format")
  target_compile_options(${KERNEL} PUBLIC "-Wno-ignored-attributes")
endif()
if(RUNHOST)
  target_compile_definitions(${KERNEL} PRIVATE RUNHOST=1)
endif()
if(INORD)
  target_compile_definitions(${KERNEL} PRIVATE INORD=1)
endif()
if(MEASURE_NORMAL)
  target_compile_definitions(${KERNEL} PRIVATE MEASURE_NORMAL=1)
endif()
if(VERIFY)
  target_compile_definitions(${KERNEL} PRIVATE VERIFY=1)
endif()
if(USE_BOOST)
  target_compile_definitions(${KERNEL} PRIVATE USE_BOOST_OPTPARSE=1)
endif()
if(USE_CIMG)
  target_compile_options(${KERNEL} PUBLIC "-Wno-pointer-to-int-cast")
  target_compile_definitions(${KERNEL} PRIVATE CIMG_ENABLE=1)
endif()
endfunction() # builds

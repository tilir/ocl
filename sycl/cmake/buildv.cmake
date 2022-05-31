#------------------------------------------------------------------------------
#
# CMake build system for DPC++/SYCL examples
#
# buildv function: builds kernel executable with some additional defines
# expect 2-4 arguments: target source [define] [define]
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

include(cmake/builds.cmake)

function(buildv KERNEL SRC)
  set(KSPV "${KERNEL}_SPV")
  add_executable(${KERNEL} ${SRC})
  target_compile_options(${KERNEL} PUBLIC "-fsycl" "-fsycl-unnamed-lambda")
  target_compile_features(${KERNEL} PRIVATE cxx_std_20)
  builds(${KERNEL} "${ARGV2}" "${ARGV3}")
  target_link_libraries(${KERNEL} testers-frame)
if (NOT WIN32)
  # Linux build fails on CIMG and other libs if pthread is not used
  target_link_libraries(${KERNEL} pthread)
  if(USE_CIMG)
    target_link_libraries(${KERNEL} X11)
  endif()
endif()
if(USE_BOOST)
  target_link_libraries(${KERNEL} Boost::program_options)
endif()
if(USE_CIMG)
  target_link_libraries(${KERNEL} jpeg)
endif()
if(DUMP_SPIRV)
  # object library to prevent linking
  add_library(${KSPV} OBJECT ${SRC})
  target_compile_options(${KSPV} PUBLIC "-fsycl" "-fsycl-device-only" "-fno-sycl-use-bitcode" "-fsycl-unnamed-lambda" "-o" "${KERNEL}.spv")
  target_compile_features(${KSPV} PRIVATE cxx_std_20)
  builds(${KSPV} "${ARGV2}" "${ARGV3}")

  # still link with interface library
  target_link_libraries(${KSPV} testers-frame)
endif()
endfunction() # buildv


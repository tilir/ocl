#expect 2-3 arguments: target source [define] [define]
function(buildv KERNEL SRC)
  add_executable(${KERNEL} ${SRC})
  target_compile_options(${KERNEL} PUBLIC "-fsycl" "-fsycl-unnamed-lambda")
  target_compile_features(${KERNEL} PRIVATE cxx_std_20)
  if (NOT "${ARGV2}" STREQUAL "")
    target_compile_definitions(${KERNEL} PRIVATE ${ARGV2})
  endif()
  if (NOT "${ARGV3}" STREQUAL "")
    target_compile_definitions(${KERNEL} PRIVATE ${ARGV3})
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
  target_link_libraries(${KERNEL} Boost::program_options)
endif()
  target_link_libraries(${KERNEL} testers-frame)

# under WIN32 OneAPI ditribution have Release driver, so no dumps possible through env
if(NOT WIN32)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/${KERNEL})
  add_test(NAME ${KERNEL}_run
     COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/${KERNEL}" ${CMAKE_BINARY_DIR}/${KERNEL}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
endfunction() # buildv


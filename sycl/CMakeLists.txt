# run as cmake -DCMAKE_CXX_COMPILER=${DPC_COMPILER}/clang++ ..

cmake_minimum_required(VERSION 3.14)

project(syclexamples)

set(Boost_USE_MULTITHREADED ON)

# Ugliest hack possible for stable Windows builds
# Idea is: we do not have regular linker syntax:
# 'dpcpp target.obj /link <linker args>' works, but 'dpcpp target.obj <linker args>' not.
# So we have to redefine CMAKE_CXX_LINK_EXECUTABLE target, which is strongly disregarded of course.
# Madness, freaking madness.
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc -Wno-ignored-attributes")
  set(CMAKE_EXE_LINKER_FLAGS "-fsycl")
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /nologo <DEFINES> <INCLUDES> /EHsc <FLAGS> /Fo<OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib /nologo <OBJECTS> /out:<TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl /nologo <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

find_package(Boost COMPONENTS
             program_options
             # other if required
             REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(KERNELS
  vectoradd
  vectoradd_complexdeps
  vectoradd_devicemem
  vectoradd_inord
  vectoradd_sharedmem
  vectoradd_wait
  vectoradd_inequal
  vectoradd_simd
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps)

# common includes and libs
foreach(KERNEL ${KERNELS})
  add_executable(${KERNEL} ${KERNEL}.cc)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/${KERNEL})
  target_compile_options(${KERNEL} PUBLIC "-fsycl" "-fsycl-unnamed-lambda")
  target_compile_features(${KERNEL} PRIVATE cxx_std_20)
  target_link_libraries(${KERNEL} Boost::program_options)

# under WIN32 OneAPI ditribution have Release driver, so no dumps possible
if(NOT WIN32)
  add_test(NAME ${KERNEL}_run
     COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/${KERNEL}" ${CMAKE_BINARY_DIR}/${KERNEL}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
endforeach()

# special executable for host mem allocation
add_executable(vectoradd_hostmem vectoradd_sharedmem.cc)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/vectoradd_hostmem)
target_compile_options(vectoradd_hostmem PUBLIC "-fsycl" "-fsycl-unnamed-lambda" "-DHOST_ALLOC")
target_compile_features(vectoradd_hostmem PRIVATE cxx_std_20)
target_link_libraries(vectoradd_hostmem Boost::program_options)

# under WIN32 OneAPI ditribution have Release driver, so no dumps possible
if(NOT WIN32)
add_test(NAME vectoradd_hostmem_run
   COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/vectoradd_hostmem" ${CMAKE_BINARY_DIR}/vectoradd_hostmem
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
endif()

# dummy install
install()
enable_testing()

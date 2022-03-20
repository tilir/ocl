# on linux (or windows with unix makefiles)
# run as: cmake -DCMAKE_CXX_COMPILER=${DPC_COMPILER}/clang++ ..
#
# on windows it is like
# with Visual Studio: cmake -T "Intel(R) oneAPI DPC++ Compiler" ..
# with Ninja: cmake -GNinja -DCMAKE_CXX_COMPILER=dpcpp ..

cmake_minimum_required(VERSION 3.21)
project(syclexamples)

if(USE_BOOST)
set(Boost_USE_MULTITHREADED ON)
endif()

# Ugliest hack possible for stable Windows builds
# Idea is: we do not have regular linker syntax:
# 'dpcpp target.obj /link <linker args>' works, but 'dpcpp target.obj <linker args>' not.
# So we have to redefine CMAKE_CXX_LINK_EXECUTABLE target, which is strongly disregarded of course.
# Madness, freaking madness.
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc")
  set(CMAKE_EXE_LINKER_FLAGS "-fsycl")
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /nologo <DEFINES> <INCLUDES> /EHsc <FLAGS> /Fo<OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib /nologo <OBJECTS> /out:<TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl /nologo <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

#find_package(Intel_DPCPP REQUIRED)

if(USE_BOOST)
find_package(Boost COMPONENTS
             program_options
             # other if required
             REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
endif()

#expect 3 arguments: target source define
function(buildv KERNEL SRC)
  add_executable(${KERNEL} ${SRC})
  target_compile_options(${KERNEL} PUBLIC "-fsycl" "-fsycl-unnamed-lambda")
  target_compile_features(${KERNEL} PRIVATE cxx_std_20)
  target_compile_definitions(${KERNEL} PRIVATE ${ARGV2})
if(USE_BOOST)
  target_link_libraries(${KERNEL} Boost::program_options)
endif()

# under WIN32 OneAPI ditribution have Release driver, so no dumps possible through env
if(NOT WIN32)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps/${KERNEL})
  add_test(NAME ${KERNEL}_run
     COMMAND ${CMAKE_COMMAND} -E env "IGC_ShaderDumpEnable=1" env "IGC_ShaderDumpEnableAll=1" env "IGC_DumpToCustomDir=${CMAKE_BINARY_DIR}/dumps/${KERNEL}" ${CMAKE_BINARY_DIR}/${KERNEL}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
endfunction() # buildv

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

if(NOT WIN32)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps)
endif()

# common includes and libs
foreach(KERNEL ${KERNELS})
  buildv(${KERNEL} ${KERNEL}.cc)
endforeach()

# special target with define
buildv(vectoradd_hostmem vectoradd_sharedmem.cc "HOST_ALLOC=1")

# dummy install
install()
enable_testing()
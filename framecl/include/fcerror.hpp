//-----------------------------------------------------------------------------
//
// framecl utilities: options, timer, etc
//
//-----------------------------------------------------------------------------

// clang-format off
#include "cldefs.h"
#include "CL/opencl.hpp"
// clang-format on

struct process_error_t {
  cl_int ret;

  process_error_t(cl_int r) : ret(r) {}

  void operator()(std::ostream &os) {
    const char *cause = "unknown";

    if (ret == CL_SUCCESS) {
      return;
    }
#undef PROCESS
#define PROCESS(ret, STR)                                                      \
  else if (ret == STR) {                                                       \
    cause = #STR;                                                              \
  }
    PROCESS(ret, CL_DEVICE_NOT_FOUND)
    PROCESS(ret, CL_DEVICE_NOT_AVAILABLE)
    PROCESS(ret, CL_COMPILER_NOT_AVAILABLE)
    PROCESS(ret, CL_MEM_OBJECT_ALLOCATION_FAILURE)
    PROCESS(ret, CL_OUT_OF_RESOURCES)
    PROCESS(ret, CL_OUT_OF_HOST_MEMORY)
    PROCESS(ret, CL_OUT_OF_HOST_MEMORY)
    PROCESS(ret, CL_PROFILING_INFO_NOT_AVAILABLE)
    PROCESS(ret, CL_MEM_COPY_OVERLAP)
    PROCESS(ret, CL_IMAGE_FORMAT_MISMATCH)
    PROCESS(ret, CL_IMAGE_FORMAT_NOT_SUPPORTED)
    PROCESS(ret, CL_BUILD_PROGRAM_FAILURE)
    PROCESS(ret, CL_MAP_FAILURE)
    PROCESS(ret, CL_MISALIGNED_SUB_BUFFER_OFFSET)
    PROCESS(ret, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    PROCESS(ret, CL_COMPILE_PROGRAM_FAILURE)
    PROCESS(ret, CL_LINKER_NOT_AVAILABLE)
    PROCESS(ret, CL_LINK_PROGRAM_FAILURE)
    PROCESS(ret, CL_DEVICE_PARTITION_FAILED)
    PROCESS(ret, CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
    PROCESS(ret, CL_INVALID_VALUE)
    PROCESS(ret, CL_INVALID_DEVICE_TYPE)
    PROCESS(ret, CL_INVALID_PLATFORM)
    PROCESS(ret, CL_INVALID_DEVICE)
    PROCESS(ret, CL_INVALID_VALUE)
    PROCESS(ret, CL_INVALID_DEVICE_TYPE)
    PROCESS(ret, CL_INVALID_PLATFORM)
    PROCESS(ret, CL_INVALID_DEVICE)
    PROCESS(ret, CL_INVALID_CONTEXT)
    PROCESS(ret, CL_INVALID_QUEUE_PROPERTIES)
    PROCESS(ret, CL_INVALID_COMMAND_QUEUE)
    PROCESS(ret, CL_INVALID_HOST_PTR)
    PROCESS(ret, CL_INVALID_MEM_OBJECT)
    PROCESS(ret, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    PROCESS(ret, CL_INVALID_IMAGE_SIZE)
    PROCESS(ret, CL_INVALID_SAMPLER)
    PROCESS(ret, CL_INVALID_BINARY)
    PROCESS(ret, CL_INVALID_BUILD_OPTIONS)
    PROCESS(ret, CL_INVALID_PROGRAM)
    PROCESS(ret, CL_INVALID_PROGRAM_EXECUTABLE)
    PROCESS(ret, CL_INVALID_KERNEL_NAME)
    PROCESS(ret, CL_INVALID_KERNEL_DEFINITION)
    PROCESS(ret, CL_INVALID_KERNEL)
    PROCESS(ret, CL_INVALID_ARG_INDEX)
    PROCESS(ret, CL_INVALID_ARG_VALUE)
    PROCESS(ret, CL_INVALID_ARG_SIZE)
    PROCESS(ret, CL_INVALID_KERNEL_ARGS)
    PROCESS(ret, CL_INVALID_WORK_DIMENSION)
    PROCESS(ret, CL_INVALID_WORK_GROUP_SIZE)
    PROCESS(ret, CL_INVALID_WORK_ITEM_SIZE)
    PROCESS(ret, CL_INVALID_GLOBAL_OFFSET)
    PROCESS(ret, CL_INVALID_EVENT_WAIT_LIST)
    PROCESS(ret, CL_INVALID_EVENT)
    PROCESS(ret, CL_INVALID_OPERATION)
    PROCESS(ret, CL_INVALID_GL_OBJECT)
    PROCESS(ret, CL_INVALID_BUFFER_SIZE)
    PROCESS(ret, CL_INVALID_MIP_LEVEL)
    PROCESS(ret, CL_INVALID_GLOBAL_WORK_SIZE)
    PROCESS(ret, CL_INVALID_PROPERTY)
    PROCESS(ret, CL_INVALID_IMAGE_DESCRIPTOR)
    PROCESS(ret, CL_INVALID_COMPILER_OPTIONS)
    PROCESS(ret, CL_INVALID_LINKER_OPTIONS)
    PROCESS(ret, CL_INVALID_DEVICE_PARTITION_COUNT)
#if (CL_TARGET_OPENCL_VERSION > 120)
    PROCESS(ret, CL_INVALID_PIPE_SIZE)
    PROCESS(ret, CL_INVALID_DEVICE_QUEUE)
#endif
#undef PROCESS
    else {
      cause = "Unknown";
    }

    os << "Error: <" << cause << "> code = " << ret;
  }
};

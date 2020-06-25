## FrameCL reference

### Basic information

FrameCL is simple framework to make in-depth OpenCL programming experiments

Just compile your program with it and you will get for free:

* Classes
  * Utility classes to work with options
  * Core classes to wrap basic concepts
  * Platform and device enumerations
  * Execution timings
* Dependency graphs
  * Declarative markup of tasks
  * Automatical event management for out of order queues

### Depgraph example

Format is:

    { task_ptr [required tasks] }

Simple vector addition dependency graph

    framecl::depgraph_t dg(ctx, {
      {&writeA},
      {&writeB},
      {&execF, &writeA, &writeB},
      {&readC, &execF}
    });

execution depends on writing to devices buffers A and B, and reading buffer C depends on execution

    dg.execute()

will execute whole graph on given context

### Config & build

How to build in release mode and run simple e2e tests:

    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CIMG_KERNELS=OFF ../ocl/framecl/
    cmake --build . -j
    cmake --build . --target check -j

Note: to run tests you need to have both Intel and NVIDIA GPU's (like Intel integrated and NVIDIA discrete or so)
Use `-DINCLUDE_INTEL_TESTS=OFF` or `-DINCLUDE_NVIDIA_TESTS=OFF` to exclude part of those
Use `-DINCLUDE_FAILING_TESTS=ON` to include tests that are still in progress of debugging

Some kernels are using CImg library. You may put option `-DBUILD_CIMG_KERNELS=ON` (default) to build those, but then you may need to specify images and have at least X11 and JPEG cmake packages available


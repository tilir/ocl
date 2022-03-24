## SYCL experiments

### Directory structure

* cmake for cmake modules
* framework for common utilities
* txt for textual notes
* named folders for different kernels
  * vadd for vector add and friends
  * sgemm for matrix multiplications
  * bitonic for bitonic sorts

### Config & build

On linux (or windows with unix makefiles)

    cmake -DCMAKE_CXX_COMPILER=${DPC_COMPILER}/clang++ ..

On windows it is like

with Visual Studio: 

    cmake -T "Intel(R) oneAPI DPC++ Compiler" ..

with Ninja:

    cmake -GNinja -DCMAKE_CXX_COMPILER=dpcpp ..

Note that on Windows you may need cmake > 3.20

### Framework reference

Framework makes it easy to write files solely with kernel code, abstracting away all boring stuff like option parsing, test sequences, platform initialization, etc.

Start with <xxx>_testers file, including testers.hpp

You need to define common interface like:

    template <typename T> class BitonicSort 

In this class what is required is operator() signature
For bitonic sort it is:

    virtual EvtRet_t operator()(T *Vec, size_t Sz) = 0;

For sgemm it is:

    virtual EvtRet_t operator()(const T *A, const T *B, T *C, size_t AX,
                                size_t AY, size_t BY) = 0;

Now you need to have somewhere two classes for CPU and for GPU

CPU (or host) example:

    template <typename T> 
    struct BitonicSortHost : public BitonicSort<T>

GPU example:

    template <typename T>
    class BitonicSortBuf : public sycltesters::BitonicSort<T>

Now all you need is:
  * define test sequence
  * write code for this or that kernel variant for GPU

See a ton of different vector add examples.

To use test sequence 

    sycltesters::test_sequence<BitonicSortBuf<int>>(argc, argv);

Inside your main function.

Now you are ready to go, and it is much simpler to read and present than full-featured programs. Every file like bitonicsort.cc, matmult.cc, vectoradd.cc now contains only essential SYCL kernel, nothing duplicating.

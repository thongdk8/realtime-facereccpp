## Download and extract clang+ compiler 6.0
Go to [this link](http://releases.llvm.org/download.html) and select llvm 6.0 which supported your os. For example [Ubuntu 16.04](http://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz)

## Install TVM with clang supported
    git clone --recursive https://github.com/dmlc/tvm
    git submodule init
    git submodule update
    mkdir build
    cp cmake/config.cmake build
Edit build/config.cmake to customize the compilation options.

Change set(USE_CUDA OFF) to set(USE_CUDA ON) to enable CUDA backend. So do other backends and libraries (OpenCL, RCOM, METAL, VULKAN, …).

Modify build/config.cmake to add set(USE_LLVM  **/path/to/your/llvm/bin/llvm-config**). Remember that we have downloaded clang 6.0 in previous step.

    cd build
    cmake ..
    make -j4
    sudo make install
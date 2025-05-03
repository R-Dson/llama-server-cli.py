#!/bin/bash

# Add more targets if needed
BUILD_TARGET="llama-server"
# Change this if needed
FINAL_BUILD_TARGET_PATH="../"

LLAMA_CPP_DIR="./llama.cpp"
CCACHE_EXISTS=$(command -v ccache &> /dev/null && echo "true" || echo "false")
# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi could not be found. This is intended to be used with NVIDIA GPUs."
    exit 1
fi

# Check if we are in the "llama-server-help-script" folder
current_dir=$(basename "$(pwd)")
if [ "$current_dir" = "llama-server-help-script" ]; then
    cd ..
fi

# Clone or pull llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
    cd "$LLAMA_CPP_DIR"
else
    cd "$LLAMA_CPP_DIR"
    git pull
fi

# Configure with CMake
if [ "$CCACHE_EXISTS" = "true" ]; then
    cmake . -B ./build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_COMPILER_LAUNCHER="ccache" -DCMAKE_C_COMPILER_LAUNCHER="ccache" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache"
else
    cmake . -B ./build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
fi

# Compile
NUM_THREADS=$(nproc)
nice cmake --build ./build --config Release -j $NUM_THREADS --clean-first --target $BUILD_TARGET

# Print version
if [[ "${BUILD_TARGET,,}" == *"llama-server"* ]]; then
    ./build/bin/$BUILD_TARGET --version | grep 'version: '
fi

# Move binary
mv "./build/bin/$BUILD_TARGET" "$FINAL_BUILD_TARGET_PATH/$BUILD_TARGET"

# Clean up
# rm -rf ./build

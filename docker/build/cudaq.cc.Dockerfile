# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:llvm-main
FROM $base_image

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"

RUN mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD \
    && mkdir build-compiler-rt && cd build-compiler-rt \
    && cmake -G Ninja ../compiler-rt -DCMAKE_C_COMPILER=/opt/llvm/bootstrap/cc -DCMAKE_CXX_COMPILER=/opt/llvm/bootstrap/cxx -DLLVM_CONFIG_PATH=/opt/llvm/bin/llvm-config -DCOMPILER_RT_BUILD_BUILTINS=OFF -DCOMPILER_RT_BUILD_LIBFUZZER=OFF -DCOMPILER_RT_BUILD_MEMPROF=OFF -DCOMPILER_RT_BUILD_PROFILE=ON -DCOMPILER_RT_BUILD_SANITIZERS=OFF -DCOMPILER_RT_BUILD_XRAY=OFF \
    && ninja && mkdir -p /opt/llvm/lib/clang/16/lib/linux \
    && cp ./lib/linux/libclang_rt.profile-x86_64.a /opt/llvm/lib/clang/16/lib/linux
    # && cd /llvm-project && mkdir build-tools && cd build-tools \
    # && cmake -G Ninja ../llvm -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_DISTRIBUTION_COMPONENTS="llvm-cov;llvm-profdata" -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_OPTIMIZED_TABLEGEN=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=OFF -DCMAKE_BUILD_TYPE=Release \
    # && ninja && cp ./bin/llvm-cov /opt/llvm/bin \
    # && cp ./bin/llvm-profdata /opt/llvm/bin
RUN apt update && apt install -y llvm-15 llvm-15-tools

WORKDIR "$destination"
# Configuring a base image that contains the necessary dependencies for GPU
# accelerated components and passing a build argument 
#   install="CMAKE_BUILD_TYPE=Release FORCE_COMPILE_GPU_COMPONENTS=true"
# creates a dev image that can be used as argument to docker/release/cudaq.Dockerfile
# to create the released cuda-quantum image.
ARG install=
RUN if [ -n "$install" ]; \
    then \
        expected_prefix=$CUDAQ_INSTALL_PREFIX; \
        install=`echo $install | xargs` && export $install; \
        bash scripts/build_cudaq.sh -v; \
        if [ ! "$?" -eq "0" ]; then \
            exit 1; \
        elif [ "$CUDAQ_INSTALL_PREFIX" != "$expected_prefix" ]; then \
            mkdir -p "$expected_prefix"; \
            mv "$CUDAQ_INSTALL_PREFIX"/* "$expected_prefix"; \
            rmdir "$CUDAQ_INSTALL_PREFIX"; \
        fi \
    fi

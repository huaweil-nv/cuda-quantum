#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/generate_cc.sh -c
# -or-
# bash scripts/generate_cc.sh -p
# -or-
# bash scripts/generate_cc.sh -c -p
# -c flag generates coverage information for C and C++ codes.
# -p flag generates coverage information for Python codes.
# -v flag generates data format for uploading to codecov
# C and C++ coverage reports are generated in the directory 'build/ccoverage'
# Python coverage reports are generated in the directory 'build/pycoverage'
#
# Note:
# The script should be run in the cuda-quantum-devdeps container environment.
# current tested image: ghcr.io/nvidia/cuda-quantum-devdeps:clang16-main
# Don't enable GPU
# C/C++ coverage is located in the ./build/ccoverage directory
# Python coverage is located in the ./build/pycoverage directory

if [ $# -lt 1 ]; then
    echo "Please provide at least one parameter"
    exit 1
fi

gen_cpp_coverage=false
gen_py_coverage=false
is_codecov_format=false

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":cpv" opt; do
    case $opt in
    c)
        gen_cpp_coverage=true
        ;;
    p)
        gen_py_coverage=true
        ;;
    v)
        is_codecov_format=true
        ;;
    \?)
        echo "Invalid command line option -$OPTARG" >&2
        exit 1
        ;;
    esac
done
OPTIND=$__optind__

# Repo root
this_file_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)

# Set envs
if $gen_cpp_coverage; then
    export CUDAQ_ENABLE_CC=ON
    mkdir -p /usr/lib/llvm-16/lib/clang/16/lib/linux
    ln -s /usr/local/llvm/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.profile.a /usr/lib/llvm-16/lib/clang/16/lib/linux/libclang_rt.profile-x86_64.a
    # export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-%9m.profraw
fi

# Build project
# debug
export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-build-%9m.profraw
CUDAQ_WERROR=OFF bash ${repo_root}/scripts/build_cudaq.sh
if [ $? -ne 0 ]; then
    echo "Build cudaq failure: $?" >&2
    exit 1
fi

# Function to run the llvm-cov command
gen_cplusplus_report() {
    if $is_codecov_format; then
        mkdir -p ${repo_root}/build/ccoverage
        llvm-cov show ${objects} -instr-profile=${repo_root}/build/coverage.profdata --ignore-filename-regex="${repo_root}/tpls/*" \
            --ignore-filename-regex="${repo_root}/build/*" --ignore-filename-regex="${repo_root}/unittests/*" 2>&1 > ${repo_root}/build/ccoverage/coverage.txt
    else
        llvm-cov show -format=html ${objects} -instr-profile=${repo_root}/build/coverage.profdata --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/default/rest_serve/*" --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/fermioniq/*" --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/orca/*" --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/quera/*" --ignore-filename-regex="${repo_root}/tpls/*" \
            --ignore-filename-regex="${repo_root}/build/*" --ignore-filename-regex="${repo_root}/unittests/*" --ignore-filename-regex="usr/local/cuda-13.0/*" --ignore-filename-regex="usr/local/llvm/*" -o ${repo_root}/build/ccoverage 2>&1
    fi
}

if $gen_cpp_coverage; then
    use_llvm_cov=true

    # Run tests (C++ Unittests)
    python3 -m pip install iqm-client==28.0.0
    # debug
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-ctest-%9m.profraw
    ctest --output-on-failure --test-dir ${repo_root}/build -E ctest-nvqpp
    ctest_status=$?
    # mpi tests
    # Set MPI_PATH depending on OMPI/MPICH
    # has_ompiinfo=$(which ompi_info || true)
    # if [[ ! -z $has_ompiinfo ]]; then
    #   export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/openmpi/"
    # else
    #   export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/mpich/"
    # fi
    MPI_PATH=/usr/local/openmpi
    # Run the activation script
    cd ${repo_root}/runtime/cudaq/distributed/builtin/
    cp ../distributed_capi.h .
    bash activate_custom_mpi.sh
    external_plugin_build_status=$?
    cd -
    export CUDAQ_MPI_COMM_LIB=${repo_root}/runtime/cudaq/distributed/builtin/libcudaq_distributed_interface_mpi.so
    if [ ! $external_plugin_build_status -eq 0 ] ; then
      echo "Test CUDA Quantum MPI Plugin Activation failed to activate the plugin with status $external_plugin_build_status."
    #   exit 1
    fi
    echo $CUDAQ_MPI_COMM_LIB
    # Rerun the MPI plugin test
    cd ${repo_root}
    ctest --test-dir build -R MPIApiTest -V
    external_plugin_status=$?   
    if [ ! $external_plugin_status -eq 0 ] ; then
      echo "Test CUDA Quantum MPI Plugin Activation failed with status $external_plugin_status."
    #   exit 1
    fi

    # debug
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-llvmlit-%9m.profraw
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/test/lit.site.cfg.py ${repo_root}/build/test
    lit_status=$?
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/targettests/lit.site.cfg.py ${repo_root}/build/targettests
    targ_status=$?
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/python/tests/mlir/lit.site.cfg.py ${repo_root}/build/python/tests/mlir
    pymlir_status=$?
    #if [ ! $ctest_status -eq 0 ] || [ ! $lit_status -eq 0 ] || [ $targ_status -ne 0 ] || [ $pymlir_status -ne 0 ]; then
    #    echo "::error C++ tests failed (ctest status $ctest_status, llvm-lit status $lit_status, \
    #target tests status $targ_status, Python MLIR status $pymlir_status)."
    #    exit 1
    #fi

    # Run tests (Python tests)
    rm -rf ${repo_root}/_skbuild
    pip install ${repo_root} --user -vvv
    # debug
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-python-%9m.profraw
    python3 -m pytest -v ${repo_root}/python/tests/ --ignore ${repo_root}/python/tests/backends
    for backendTest in ${repo_root}/python/tests/backends/*.py; do
        python3 -m pytest -v $backendTest
        pytest_status=$?
        if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
            echo "::error $backendTest tests failed with status $pytest_status."
            exit 1
        fi
    done

    # Generate report
    if $use_llvm_cov; then
        llvm-profdata merge -sparse ${repo_root}/build/tmp/cudaq-cc/profile-*.profraw -o ${repo_root}/build/coverage.profdata
        binarys=($(sed -n -e '/Linking CXX shared library/s/^.*Linking CXX shared library //p' \
            -e '/Linking CXX static library/s/^.*Linking CXX static library //p' \
            -e '/Linking CXX shared module/s/^.*Linking CXX shared module //p' \
            -e '/Linking CXX executable/s/^.*Linking CXX executable //p' ${repo_root}/build/logs/ninja_output.txt))
        objects=""
        for item in "${binarys[@]}"; do
            objects+="-object ${repo_root}/build/$item "
        done

        # The purpose of adding this code is to avoid the llvm-cov show command
        # from being unable to generate a report due to a malformed format error of an object.
        # This is mainly an error caused by a static library, but it has little impact on the coverage rate.
        # Loop until the command succeeds
        while true; do
            output=$(gen_cplusplus_report ${objects})
            status=$?

            # Check if the command failed due to malformed coverage data
            if [ $status -ne 0 ]; then
                echo "Error detected. Attempting to remove problematic object and retry."
                echo "$output"

                # Extract the problematic object from the error message
                problematic_object=$(echo "$output" | grep -oP "error: Failed to load coverage: '\K[^']+")
                echo $problematic_object

                if [ -n "$problematic_object" ]; then
                    # Remove the problematic object from the objects variable
                    objects=$(echo $objects | sed "s|-object $problematic_object||")

                    # Check if the problematic object was successfully removed
                    if [[ $objects != *"-object $problematic_object"* ]]; then
                        echo "Problematic object '$problematic_object' removed. Retrying..."
                    else
                        echo "Failed to remove problematic object '$problematic_object'. Exiting..."
                        exit 1
                    fi
                else
                    echo "No problematic object found in the error message. Exiting..."
                    exit 1
                fi
            else
                echo "Command succeeded."
                break
            fi
        done
    else
        # Use gcov
        echo "Currently not supported, running tests using llvm-lit fails"
        exit 1
    fi
fi

if $gen_py_coverage; then
    pip install coverage
    pip install iqm_client==28.0.0 --user -vvv
    rm -rf ${repo_root}/_skbuild
    pip install -e . --user -vvv

    # normal tests
    coverage run -a -m pytest -v python/tests/ --ignore python/tests/backends
    # backend tests
    for backendTest in python/tests/backends/*.py; do
        coverage run -a -m pytest -v $backendTest
        pytest_status=$?
        if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
            echo "::error $backendTest tests failed with status $pytest_status."
            exit 1
        fi
    done
    # mlir tests
    # Iterate through all .py files in python/tests/mlir directory (including subdirectories)
    find ${repo_root}/python/tests/mlir -name "*.py" | while read -r test_file; do
        # Check if the file contains XFAIL marker, if so skip it
        if grep -q "# XFAIL:" "$test_file"; then
            echo "Skipping file with XFAIL marker: $test_file"
            continue
        fi
        
        # Check if the file contains RUN instruction
        run_line=$(grep "# RUN:" "$test_file" | head -n 1)
        if [ -z "$run_line" ]; then
            echo "Skipping file without RUN instruction: $test_file"
            continue
        fi
        
        # Determine how to run the test based on RUN instruction
        if [[ "$run_line" == *"python"* ]]; then
            echo "Running test with python: $test_file"
            coverage run -a "$test_file"
        elif [[ "$run_line" == *"pytest"* ]]; then
            echo "Running test with pytest: $test_file"
            coverage run -a -m pytest "$test_file"
        else
            echo "RUN instruction format unclear, skipping: $test_file"
        fi
    done

    # generate report
    if $is_codecov_format; then
        coverage xml -o ${repo_root}/build/pycoverage/coverage.xml --omit=${repo_root}/python/tests/*,/usr/lib/*
    else
        coverage html -d ${repo_root}/build/pycoverage --omit=${repo_root}/python/tests/*,/usr/lib/*
    fi
fi

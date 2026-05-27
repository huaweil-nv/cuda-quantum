#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
# current tested image: ghcr.io/nvidia/cuda-quantum-devdeps:llvm-main
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
fi

# Build project
# debug
export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-build-%9m.profraw
# CUDAQ_WERROR=OFF bash ${repo_root}/scripts/build_cudaq.sh -- -DCUDAQ_TEST_OMP_SLOTS=2
bash ${repo_root}/scripts/build_cudaq.sh -- -DCUDAQ_TEST_OMP_SLOTS=2
if [ $? -ne 0 ]; then
    echo "Build cudaq failure: $?" >&2
    exit 1
fi

# Functions to run llvm-cov commands.
set_cplusplus_ignore_args() {
    coverage_ignore_args=(
        --ignore-filename-regex="${repo_root}/tpls/*"
        --ignore-filename-regex="${repo_root}/build/*"
        --ignore-filename-regex="${repo_root}/unittests/*"
        --ignore-filename-regex="usr/include/python[0-9.]*/.*"
    )

    if ! $is_codecov_format; then
        coverage_ignore_args+=(
            --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/default/rest_serve/*"
            --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/fermioniq/*"
            --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/orca/*"
            --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/quera/*"
            --ignore-filename-regex="usr/local/cuda-13.0/*"
            --ignore-filename-regex="usr/local/llvm/*"
            --ignore-filename-regex="${repo_root}/python/tests/interop/test_cpp_quantum_algorithm_module.cpp"
            --ignore-filename-regex="${repo_root}/runtime/test/test_argument_conversion.cpp"
            --ignore-filename-regex="${repo_root}/runtime/cudaq/platform/default/rest/helpers/braket/*"
            --ignore-filename-regex="${repo_root}/runtime/common/Braket.*"
        )
    fi
}

gen_cplusplus_report() {
    mkdir -p ${repo_root}/build/ccoverage
    if $is_codecov_format; then
        llvm-cov show ${objects} -instr-profile=${repo_root}/build/coverage.profdata \
            "${coverage_ignore_args[@]}" 2>&1 > ${repo_root}/build/ccoverage/coverage.txt
    else
        llvm-cov show -format=html ${objects} -instr-profile=${repo_root}/build/coverage.profdata \
            "${coverage_ignore_args[@]}" -o ${repo_root}/build/ccoverage 2>&1
    fi
}

gen_cplusplus_export() {
    mkdir -p ${repo_root}/build/ccoverage
    llvm-cov export ${objects} -instr-profile=${repo_root}/build/coverage.profdata \
        "${coverage_ignore_args[@]}" > ${repo_root}/build/ccoverage/coverage.json
}

if $gen_cpp_coverage; then
    use_llvm_cov=true

    # Run CTest and lit suites through the same wrapper used by CI.
    python3 -m pip install -r ${repo_root}/requirements-tests-backend.txt --break-system-packages
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-run-tests-%9m.profraw
    bash ${repo_root}/scripts/run_tests.sh -v -B ${repo_root}/build
    run_tests_status=$?
    if [ ! $run_tests_status -eq 0 ]; then
      echo "::error run_tests.sh failed with status $run_tests_status."
      exit 1
    fi

    # Run the custom MPI plugin activation test the same way CI does.
    has_ompiinfo=$(which ompi_info || true)
    if [[ ! -z $has_ompiinfo ]]; then
      export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/openmpi/"
    else
      export MPI_PATH="/usr/lib/$(uname -m)-linux-gnu/mpich/"
    fi
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-mpi-plugin-%9m.profraw
    cd ${repo_root}/runtime/cudaq/distributed/builtin/
    cp ../distributed_capi.h .
    source activate_custom_mpi.sh
    external_plugin_build_status=$?
    cd -
    if [ ! $external_plugin_build_status -eq 0 ] ; then
      echo "Test CUDA Quantum MPI Plugin Activation failed to activate the plugin with status $external_plugin_build_status."
      exit 1
    fi
    echo $CUDAQ_MPI_COMM_LIB
    # Rerun the MPI plugin test
    cd ${repo_root}
    ctest --test-dir build -R MPIApiTest -V
    external_plugin_status=$?   
    if [ ! $external_plugin_status -eq 0 ] ; then
      echo "Test CUDA Quantum MPI Plugin Activation failed with status $external_plugin_status."
      exit 1
    fi

    # Run tests (Python tests)
    rm -rf ${repo_root}/_skbuild
    pip install ${repo_root} --user -vvv
    # debug
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-python-%9m.profraw
    python3 -m pytest -v ${repo_root}/python/tests/ \
        --ignore ${repo_root}/python/tests/backends \
        --ignore ${repo_root}/python/tests/contrib \
        --ignore ${repo_root}/python/tests/parallel
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ]; then
        echo "::error Python tests failed with status $pytest_status."
        exit 1
    fi
    for backendTest in ${repo_root}/python/tests/backends/*.py; do
        python3 -m pytest -v $backendTest
        pytest_status=$?
        if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
            echo "::error $backendTest tests failed with status $pytest_status."
            exit 1
        fi
    done
    python3 -m pip install qiskit --user
    qiskit_status=$?
    if [ ! $qiskit_status -eq 0 ]; then
        echo "::error qiskit installation failed with status $qiskit_status."
        exit 1
    fi
    python3 -m pytest -v ${repo_root}/python/tests/contrib
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ]; then
        echo "::error Python contrib tests failed with status $pytest_status."
        exit 1
    fi

    # Generate report
    if $use_llvm_cov; then
        llvm-profdata merge -sparse ${repo_root}/build/tmp/cudaq-cc/profile-*.profraw -o ${repo_root}/build/coverage.profdata
        binarys=($(sed -n -e '/Linking CXX shared library/s/^.*Linking CXX shared library //p' \
            -e '/Linking CXX static library/s/^.*Linking CXX static library //p' \
            -e '/Linking CXX shared module/s/^.*Linking CXX shared module //p' \
            -e '/Linking CXX executable/s/^.*Linking CXX executable //p' ${repo_root}/build/logs/ninja_output.txt))
        objects=""
        for item in "${binarys[@]}"; do
            # Static libraries (.a) often produce malformed coverage data; only use shared libs and executables
            [[ "$item" == *.a ]] && continue
            objects+="-object ${repo_root}/build/$item "
        done
        set_cplusplus_ignore_args

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
        gen_cplusplus_export
        export_status=$?
        if [ ! $export_status -eq 0 ]; then
            echo "::error llvm-cov export failed with status $export_status."
            exit 1
        fi
    else
        # Use gcov
        echo "Currently not supported, running tests using llvm-lit fails"
        exit 1
    fi
fi

if $gen_py_coverage; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    apt install -y python${PY_VER}-venv
 
    # Needs to be installed outside of venv
    python3 -m pip install -r ${repo_root}/requirements-tests-backend.txt --break-system-packages

    venv_dir=${repo_root}/build/venv-coverage
    python3 -m venv "$venv_dir"
    . "${venv_dir}/bin/activate"
    pip install coverage
    rm -rf ${repo_root}/_skbuild
    pip install -e . -vvv
    mkdir -p ${repo_root}/build/pycoverage
    coverage run -a -m pytest -v python/tests/ \
        --ignore python/tests/backends \
        --ignore python/tests/contrib \
        --ignore python/tests/parallel \
        --ignore python/tests/mlir
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ]; then
        echo "::error Python coverage tests failed with status $pytest_status."
        exit 1
    fi
    for backendTest in python/tests/backends/*.py; do
        coverage run -a -m pytest -v $backendTest
        pytest_status=$?
        if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
            echo "::error $backendTest tests failed with status $pytest_status."
            exit 1
        fi
    done
    pip install qiskit
    qiskit_status=$?
    if [ ! $qiskit_status -eq 0 ]; then
        echo "::error qiskit installation failed with status $qiskit_status."
        exit 1
    fi
    coverage run -a -m pytest -v python/tests/contrib
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ]; then
        echo "::error Python contrib tests failed with status $pytest_status."
        exit 1
    fi
    # MLIR regression tests are lit tests and depend on lit site configuration,
    # FileCheck semantics, and build-tree targets. They are covered above by
    # run_tests.sh instead of being executed directly through coverage.py.

    # generate report
    if $is_codecov_format; then
        coverage xml -o ${repo_root}/build/pycoverage/coverage.xml --omit=${repo_root}/python/tests/*,/usr/lib/*
    else
        coverage html -d ${repo_root}/build/pycoverage --omit=${repo_root}/python/tests/*,/usr/lib/*
    fi
fi

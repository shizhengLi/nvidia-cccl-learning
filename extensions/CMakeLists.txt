cmake_minimum_required(VERSION 3.18)

project(GPUStreamProcessor
    LANGUAGES CXX CUDA
    VERSION 1.0.0
)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

# 设置编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 添加CCCL路径
set(CCCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../cccl")
if(NOT EXISTS "${CCCL_ROOT}")
    message(FATAL_ERROR "CCCL not found at ${CCCL_ROOT}")
endif()

# 添加CCCL头文件路径
include_directories(
    ${CCCL_ROOT}
    ${CCCL_ROOT}/thrust
    ${CCCL_ROOT}/cub
    ${CCCL_ROOT}/libcudacxx/include
)

# 定义宏来配置CCCL
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

# 创建可执行文件
add_executable(test_gpu_stream_processor
    test_gpu_stream_processor.cu
)

# 设置CUDA编译属性
set_target_properties(test_gpu_stream_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 链接库
target_link_libraries(test_gpu_stream_processor
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# 设置包含目录
target_include_directories(test_gpu_stream_processor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 添加编译定义
target_compile_definitions(test_gpu_stream_processor PRIVATE
    _CCCL_IMPLICIT_SYSTEM_HEADER_GCC=1  # 如果使用GCC
)

# 安装规则
install(FILES
    gpu_stream_processor.hpp
    DESTINATION include/cccl_extensions
)

# 测试
enable_testing()

# 添加基本测试
add_test(NAME BasicFunctionalityTest
         COMMAND test_gpu_stream_processor)

# 添加性能测试
add_test(NAME PerformanceBenchmark
         COMMAND test_gpu_stream_processor --benchmark)

# 设置测试超时（单位：秒）
set_tests_properties(BasicFunctionalityTest PROPERTIES TIMEOUT 120)
set_tests_properties(PerformanceBenchmark PROPERTIES TIMEOUT 300)

# 打印配置信息
message(STATUS "=== GPU Stream Processor Configuration ===")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CCCL Root: ${CCCL_ROOT}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "========================================")

# 自定义目标：运行测试
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_gpu_stream_processor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running GPU Stream Processor tests"
)

# 自定义目标：代码格式化（可选）
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    add_custom_target(format_code
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i *.hpp *.cu
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting source code"
    )
endif()

# 自定义目标：内存检查（可选）
find_program(CUDA_MEMCHECK_EXECUTABLE cuda-memcheck)
if(CUDA_MEMCHECK_EXECUTABLE)
    add_custom_target(memcheck
        COMMAND ${CUDA_MEMCHECK_EXECUTABLE} ./test_gpu_stream_processor
        DEPENDS test_gpu_stream_processor
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running CUDA memory check"
    )
endif()

# 添加文档生成（可选）
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
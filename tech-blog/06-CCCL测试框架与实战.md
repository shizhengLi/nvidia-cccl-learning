# CCCL测试框架与实战：构建可靠的GPU应用

## 引言

在完成了CCCL的深入分析和功能扩展之后，构建一个全面的测试框架至关重要。GPU编程的复杂性要求我们有系统性的测试策略，涵盖功能正确性、性能基准、边界条件和错误处理等多个方面。本文将详细介绍如何为CCCL应用构建完整的测试框架。

## 测试策略概述

### 1. 测试金字塔

```
    /\
   /  \     E2E测试 (少量，高价值)
  /____\
 /      \   集成测试 (适量，中价值)
/________\  单元测试 (大量，基础价值)
```

### 2. 测试类型分类

```cpp
// 测试类型枚举
enum class TestType {
    UNIT,           // 单元测试：测试单个函数或类
    INTEGRATION,    // 集成测试：测试组件间协作
    PERFORMANCE,    // 性能测试：验证性能指标
    REGRESSION,     // 回归测试：确保不引入新bug
    STRESS,         // 压力测试：测试极限条件
    FUZZ,           // 模糊测试：随机输入测试
    BENCHMARK       // 基准测试：性能对比
};
```

## 单元测试框架

### 1. GPU单元测试基础设施

```cpp
// gpu_test_framework.hpp
#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <random>
#include <chrono>
#include <type_traits>

namespace gpu_test {

/**
 * @brief GPU测试基类，提供通用的测试工具
 */
class GPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 检查CUDA设备
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";

        // 选择设备
        cudaSetDevice(0);

        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        device_info_ = DeviceInfo{
            prop.name,
            prop.totalGlobalMem,
            prop.maxThreadsPerBlock,
            prop.maxThreadsDim[0],
            prop.warpSize,
            prop.major,
            prop.minor
        };

        // 创建测试流
        cudaStreamCreate(&test_stream_);
    }

    void TearDown() override {
        // 清理CUDA资源
        if (test_stream_) {
            cudaStreamDestroy(test_stream_);
        }
    }

    struct DeviceInfo {
        std::string name;
        size_t total_memory;
        int max_threads_per_block;
        int max_threads_dim;
        int warp_size;
        int major_version;
        int minor_version;
    };

    DeviceInfo device_info_;
    cudaStream_t test_stream_ = nullptr;

    // 辅助方法
    template<typename T>
    thrust::host_vector<T> generate_random_data(size_t size, T min_val = T(0), T max_val = T(100)) {
        thrust::host_vector<T> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(min_val, max_val);
            for (auto& val : data) {
                val = dis(gen);
            }
        } else {
            std::uniform_real_distribution<T> dis(min_val, max_val);
            for (auto& val : data) {
                val = dis(gen);
            }
        }

        return data;
    }

    template<typename T>
    bool compare_vectors(const thrust::host_vector<T>& a,
                        const thrust::host_vector<T>& b,
                        T tolerance = T(1e-6)) {
        if (a.size() != b.size()) return false;

        for (size_t i = 0; i < a.size(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::abs(a[i] - b[i]) > tolerance) return false;
            } else {
                if (a[i] != b[i]) return false;
            }
        }

        return true;
    }

    void check_cuda_error(const std::string& operation) {
        cudaError_t error = cudaGetLastError();
        ASSERT_EQ(error, cudaSuccess) << operation << " failed: " << cudaGetErrorString(error);
    }
};

/**
 * @brief 参数化测试基类
 */
template<typename T>
class TypedGPUTest : public GPUTest, public ::testing::WithParamInterface<T> {
protected:
    using value_type = T;
};

// 常用的测试类型
using TestTypes = ::testing::Types<int, float, double>;
} // namespace gpu_test
```

### 2. CCCL组件单元测试

```cpp
// test_thrust_algorithms.cpp
#include "gpu_test_framework.hpp"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

class ThrustAlgorithmTest : public gpu_test::GPUTest {};

TEST_F(ThrustAlgorithmTest, SortBasic) {
    // 准备测试数据
    const size_t N = 10000;
    auto host_data = generate_random_data<int>(N, 0, 1000);
    thrust::device_vector<int> device_data = host_data;

    // 执行排序
    thrust::sort(thrust::device, device_data.begin(), device_data.end());

    // 验证结果
    thrust::host_vector<int> sorted_data = device_data;
    std::sort(host_data.begin(), host_data.end());

    EXPECT_TRUE(compare_vectors(sorted_data, host_data));
}

TEST_F(ThrustAlgorithmTest, ReduceConsistency) {
    const size_t N = 100000;
    auto host_data = generate_random_data<float>(N, -100.0f, 100.0f);
    thrust::device_vector<float> device_data = host_data;

    // 设备端规约
    float device_result = thrust::reduce(thrust::device,
                                        device_data.begin(),
                                        device_data.end(),
                                        0.0f);

    // 主机端验证
    float host_result = std::accumulate(host_data.begin(), host_data.end(), 0.0f);

    EXPECT_NEAR(device_result, host_result, 1e-4f);
}

TEST_P(ThrustAlgorithmTest, TransformCorrectness) {
    using T = gpu_test::TypedGPUTest<T>::value_type;

    const size_t N = 1000;
    auto input_data = generate_random_data<T>(N);
    thrust::device_vector<T> device_input = input_data;
    thrust::device_vector<T> device_output(N);

    // 执行变换：f(x) = x * 2 + 1
    auto transform_op = [] __device__ (T x) { return x * T(2) + T(1); };
    thrust::transform(thrust::device, device_input.begin(), device_input.end(),
                     device_output.begin(), transform_op);

    // 验证结果
    thrust::host_vector<T> output = device_output;
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(output[i], input_data[i] * T(2) + T(1));
    }
}

INSTANTIATE_TYPED_TEST_SUITE_P(ThrustAlgorithms, ThrustAlgorithmTest, gpu_test::TestTypes);

// CUB算法测试
class CubAlgorithmTest : public gpu_test::GPUTest {};

TEST_F(CubAlgorithmTest, BlockReduceCorrectness) {
    const int BLOCK_SIZE = 256;
    const int ITEMS_PER_THREAD = 4;
    const int TOTAL_ITEMS = BLOCK_SIZE * ITEMS_PER_THREAD;

    // 生成测试数据
    auto host_data = generate_random_data<int>(TOTAL_ITEMS);
    thrust::device_vector<int> device_data = host_data;

    // 使用CUB块级规约
    thrust::device_vector<int> block_results(1);

    // 这里需要调用自定义的block reduce kernel
    // block_reduce_kernel<<<1, BLOCK_SIZE>>>(
    //     thrust::raw_pointer_cast(device_data.data()),
    //     thrust::raw_pointer_cast(block_results.data())
    // );

    cudaDeviceSynchronize();
    check_cuda_error("Block reduce kernel");

    // 验证结果
    int expected = std::accumulate(host_data.begin(), host_data.end(), 0);
    thrust::host_vector<int> result = block_results;

    EXPECT_EQ(result[0], expected);
}
```

## 集成测试框架

### 1. 端到端集成测试

```cpp
// test_integration.cpp
#include "gpu_test_framework.hpp"
#include "../extensions/gpu_stream_processor.hpp"
#include <chrono>

class GPUStreamProcessorIntegrationTest : public gpu_test::GPUTest {};

TEST_F(GPUStreamProcessorIntegrationTest, BasicPipeline) {
    using namespace cccl_extensions;

    const size_t BUFFER_SIZE = 64 * 1024;
    const int NUM_STREAMS = 3;
    const size_t TOTAL_SIZE = BUFFER_SIZE * 5;

    // 创建处理器
    auto processor = std::make_unique<GPUStreamProcessor<float>>(BUFFER_SIZE, NUM_STREAMS);

    // 设置处理函数
    processor->set_process_function(
        ProcessFunctions::make_transform_function<float>(
            [] __device__ (float x) { return x * x + 1.0f; }
        )
    );

    // 准备测试数据
    auto input_data = generate_random_data<float>(TOTAL_SIZE, -10.0f, 10.0f);

    // 执行处理
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t processed = processor->process_stream(input_data.data(), TOTAL_SIZE);
    auto end_time = std::chrono::high_resolution_clock::now();

    // 验证结果
    EXPECT_EQ(processed, TOTAL_SIZE);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_LT(duration.count(), 5000);  // 应该在5秒内完成

    // 验证性能指标
    auto metrics = processor->get_metrics();
    EXPECT_GT(metrics.get_average_throughput_mbps(), 100.0f);  // 至少100 MB/s
}

TEST_F(GPUStreamProcessorIntegrationTest, MultipleProcessors) {
    using namespace cccl_extensions;

    // 创建多个处理器
    auto transform_processor = ProcessorFactory::create_transform_processor<double>(32 * 1024, 2);
    auto sort_processor = ProcessorFactory::create_sort_processor<double>(32 * 1024, 2);

    // 设置变换函数
    transform_processor->set_process_function(
        ProcessFunctions::make_transform_function<double>(
            [] __device__ (double x) { return std::abs(x); }
        )
    );

    // 准备数据
    const size_t DATA_SIZE = 128 * 1024;
    auto input_data = generate_random_data<double>(DATA_SIZE, -1000.0, 1000.0);

    // 执行处理管道
    size_t processed1 = transform_processor->process_stream(input_data.data(), input_data.size());
    size_t processed2 = sort_processor->process_stream(input_data.data(), input_data.size());

    EXPECT_EQ(processed1, DATA_SIZE);
    EXPECT_EQ(processed2, DATA_SIZE);
}
```

### 2. 内存管理集成测试

```cpp
// test_memory_management.cpp
#include "gpu_test_framework.hpp"
#include <thrust/device_allocator.h>
#include <thrust/host_vector.h>

class MemoryManagementTest : public gpu_test::GPUTest {};

TEST_F(MemoryManagementTest, DeviceAllocatorBasics) {
    using allocator_type = thrust::device_allocator<int>;

    allocator_type alloc;

    // 测试分配
    int* ptr = alloc.allocate(1000);
    EXPECT_NE(ptr, nullptr);

    // 测试释放
    alloc.deallocate(ptr, 1000);
}

TEST_F(MemoryManagementTest, LargeMemoryAllocation) {
    // 测试大内存分配
    const size_t LARGE_SIZE = 100 * 1024 * 1024;  // 100MB

    try {
        thrust::device_vector<float> large_vec(LARGE_SIZE);
        EXPECT_EQ(large_vec.size(), LARGE_SIZE);

        // 填充数据
        thrust::sequence(thrust::device, large_vec.begin(), large_vec.end());

        // 验证部分数据
        thrust::host_vector<float> host_check(100);
        thrust::copy_n(large_vec.begin(), 100, host_check.begin());

        for (int i = 0; i < 100; ++i) {
            EXPECT_FLOAT_EQ(host_check[i], static_cast<float>(i));
        }

    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Not enough device memory for large allocation test";
    }
}

TEST_F(MemoryManagementTest, MemoryLeakDetection) {
    // 获取初始内存状态
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t initial_free = free_mem;

    {
        // 创建临时vector
        thrust::device_vector<int> temp_vec(1024 * 1024);
        thrust::sequence(temp_vec.begin(), temp_vec.end());

        // 获取使用中的内存
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_during_allocation = initial_free - free_mem;

        EXPECT_GT(used_during_allocation, 1024 * 1024 * sizeof(int));
    }  // temp_vec在此处析构

    // 检查内存是否被释放
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t final_free = free_mem;

    // 允许少量内存差异（由于内存对齐等原因）
    EXPECT_NEAR(initial_free, final_free, 1024 * 1024);  // 1MB tolerance
}
```

## 性能测试框架

### 1. 基准测试基础设施

```cpp
// performance_benchmark.hpp
#pragma once

#include "gpu_test_framework.hpp"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>

namespace benchmark {

/**
 * @brief 性能基准测试基类
 */
class Benchmark {
public:
    struct Result {
        std::string name;
        double mean_time_ms;
        double std_deviation_ms;
        double min_time_ms;
        double max_time_ms;
        double throughput_mbps;
        int iterations;
    };

    Benchmark(const std::string& name, int warmup_iterations = 3, int benchmark_iterations = 10)
        : name_(name)
        , warmup_iterations_(warmup_iterations)
        , benchmark_iterations_(benchmark_iterations) {}

    virtual ~Benchmark() = default;

    Result run() {
        std::vector<double> times;

        // 预热运行
        for (int i = 0; i < warmup_iterations_; ++i) {
            run_single_iteration();
            cudaDeviceSynchronize();
        }

        // 基准测试运行
        for (int i = 0; i < benchmark_iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            run_single_iteration();
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);  // 转换为毫秒
        }

        return compute_statistics(times);
    }

protected:
    virtual void run_single_iteration() = 0;
    virtual double compute_throughput_bytes() const = 0;

private:
    std::string name_;
    int warmup_iterations_;
    int benchmark_iterations_;

    Result compute_statistics(const std::vector<double>& times) {
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        double variance = 0.0;
        for (double time : times) {
            variance += (time - mean) * (time - mean);
        }
        variance /= times.size();
        double std_dev = std::sqrt(variance);

        auto minmax = std::minmax_element(times.begin(), times.end());

        return {
            name_,
            mean,
            std_dev,
            *minmax.first,
            *minmax.second,
            compute_throughput_bytes() / (mean * 1024.0 * 1024.0 / 1000.0),  // MB/s
            static_cast<int>(times.size())
        };
    }
};

/**
 * @brief Thrust算法基准测试
 */
template<typename Algorithm, typename T>
class ThrustBenchmark : public Benchmark {
public:
    ThrustBenchmark(const std::string& name, size_t data_size)
        : Benchmark(name)
        , data_size_(data_size)
        , device_data_(data_size) {

        // 初始化测试数据
        thrust::host_vector<T> host_data = generate_random_data<T>(data_size_);
        device_data_ = host_data;
    }

protected:
    void run_single_iteration() override {
        Algorithm()(device_data_.begin(), device_data_.end());
    }

    double compute_throughput_bytes() const override {
        return data_size_ * sizeof(T) * 2;  // 假设读写一次
    }

private:
    size_t data_size_;
    thrust::device_vector<T> device_data_;
};

// 具体的算法实现
struct SortAlgorithm {
    template<typename Iterator>
    void operator()(Iterator begin, Iterator end) {
        thrust::sort(thrust::device, begin, end);
    }
};

struct ReduceAlgorithm {
    template<typename Iterator>
    void operator()(Iterator begin, Iterator end) {
        using T = typename Iterator::value_type;
        thrust::reduce(thrust::device, begin, end, T{0});
    }
};

} // namespace benchmark
```

### 2. 性能回归测试

```cpp
// test_performance_regression.cpp
#include "performance_benchmark.hpp"
#include <fstream>
#include <sstream>

class PerformanceRegressionTest : public gpu_test::GPUTest {
protected:
    void SetUp() override {
        GPUTest::SetUp();

        // 加载基准性能数据
        load_baseline_data();
    }

    struct BaselineMetric {
        std::string name;
        double baseline_throughput_mbps;
        double tolerance_percent;
    };

    std::vector<BaselineMetric> baseline_metrics_;

    void load_baseline_data() {
        // 这里可以从文件加载基准数据，或硬编码一些值
        baseline_metrics_ = {
            {"Sort_1M_float", 2000.0, 10.0},      // 2 GB/s ± 10%
            {"Reduce_1M_float", 5000.0, 10.0},    // 5 GB/s ± 10%
            {"Transform_1M_float", 3000.0, 10.0}   // 3 GB/s ± 10%
        };
    }

    void verify_performance(const benchmark::Benchmark::Result& result) {
        for (const auto& baseline : baseline_metrics_) {
            if (result.name.find(baseline.name) != std::string::npos) {
                double deviation = std::abs(result.throughput_mbps - baseline.baseline_throughput_mbps);
                double deviation_percent = (deviation / baseline.baseline_throughput_mbps) * 100.0;

                EXPECT_LE(deviation_percent, baseline.tolerance_percent)
                    << "Performance regression detected for " << result.name
                    << ": Expected >= " << baseline.baseline_throughput_mbps << " MB/s"
                    << ", Got " << result.throughput_mbps << " MB/s"
                    << " (deviation: " << deviation_percent << "%)";

                break;
            }
        }
    }
};

TEST_F(PerformanceRegressionTest, SortPerformance) {
    const size_t DATA_SIZE = 1024 * 1024;  // 1M elements

    auto benchmark = std::make_unique<benchmark::ThrustBenchmark<benchmark::SortAlgorithm, float>>(
        "Sort_1M_float", DATA_SIZE);

    auto result = benchmark->run();

    // 记录结果
    std::cout << "Sort Performance: " << result.throughput_mbps << " MB/s" << std::endl;

    // 验证性能回归
    verify_performance(result);

    // 确保基本性能要求
    EXPECT_GT(result.throughput_mbps, 1000.0);  // 至少 1 GB/s
}

TEST_F(PerformanceRegressionTest, ReducePerformance) {
    const size_t DATA_SIZE = 1024 * 1024;  // 1M elements

    auto benchmark = std::make_unique<benchmark::ThrustBenchmark<benchmark::ReduceAlgorithm, float>>(
        "Reduce_1M_float", DATA_SIZE);

    auto result = benchmark->run();

    std::cout << "Reduce Performance: " << result.throughput_mbps << " MB/s" << std::endl;

    verify_performance(result);
    EXPECT_GT(result.throughput_mbps, 2000.0);  // 至少 2 GB/s
}
```

## 压力测试框架

### 1. 大数据量测试

```cpp
// test_stress.cpp
#include "gpu_test_framework.hpp"
#include <thread>
#include <future>

class StressTest : public gpu_test::GPUTest {};

TEST_F(StressTest, LargeDatasetProcessing) {
    const size_t LARGE_DATASET_SIZE = 100 * 1024 * 1024;  // 100M elements

    try {
        // 分配大量内存
        thrust::device_vector<float> large_data(LARGE_DATASET_SIZE);

        // 初始化数据
        thrust::sequence(thrust::device, large_data.begin(), large_data.end());

        // 执行大规模排序
        auto start = std::chrono::high_resolution_clock::now();
        thrust::sort(thrust::device, large_data.begin(), large_data.end());
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        // 验证结果
        thrust::host_vector<float> sample(1000);
        thrust::copy_n(large_data.begin(), 1000, sample.begin());

        EXPECT_TRUE(std::is_sorted(sample.begin(), sample.end()));
        EXPECT_LT(duration.count(), 300);  // 应该在5分钟内完成

        std::cout << "Large dataset sort completed in " << duration.count() << " seconds" << std::endl;

    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Not enough device memory for large dataset test";
    }
}

TEST_F(StressTest, ConcurrentStreamProcessing) {
    using namespace cccl_extensions;

    const int NUM_PROCESSORS = 4;
    const size_t BUFFER_SIZE = 1024 * 1024;

    std::vector<std::unique_ptr<GPUStreamProcessor<float>>> processors;
    std::vector<std::future<void>> futures;

    // 创建多个处理器
    for (int i = 0; i < NUM_PROCESSORS; ++i) {
        auto processor = std::make_unique<GPUStreamProcessor<float>>(BUFFER_SIZE, 2);
        processor->set_process_function(
            ProcessFunctions::make_transform_function<float>(
                [] __device__ (float x) { return sinf(x) * cosf(x); }
            )
        );
        processors.push_back(std::move(processor));
    }

    // 并发执行
    for (int i = 0; i < NUM_PROCESSORS; ++i) {
        futures.push_back(std::async(std::launch::async, [&processors, i, BUFFER_SIZE]() {
            auto data = generate_random_data<float>(BUFFER_SIZE * 3);
            processors[i]->process_stream(data.data(), data.size());
        }));
    }

    // 等待所有处理器完成
    for (auto& future : futures) {
        future.wait();
    }

    // 验证所有处理器都成功完成
    for (const auto& processor : processors) {
        auto metrics = processor->get_metrics();
        EXPECT_GT(metrics.total_processed.load(), 0);
    }
}

TEST_F(StressTest, MemoryPressure) {
    // 测试内存压力情况下的行为
    std::vector<thrust::device_vector<int>> vectors;

    try {
        // 尽可能多地分配内存
        for (int i = 0; i < 100; ++i) {
            vectors.emplace_back(10 * 1024 * 1024);  // 10M ints each

            // 执行一些操作
            thrust::sequence(vectors.back().begin(), vectors.back().end());
            thrust::sort(thrust::device, vectors.back().begin(), vectors.back().end());

            std::cout << "Allocated " << (i + 1) << " vectors successfully" << std::endl;
        }

    } catch (const std::bad_alloc&) {
        std::cout << "Memory allocation failed after " << vectors.size() << " allocations" << std::endl;
        EXPECT_GT(vectors.size(), 1);  // 至少应该能分配一个vector
    }

    // 验证已分配的vector仍然可用
    for (size_t i = 0; i < vectors.size(); ++i) {
        int sum = thrust::reduce(thrust::device, vectors[i].begin(), vectors[i].end(), 0);
        EXPECT_GT(sum, 0);
    }
}
```

## 自动化测试基础设施

### 1. 持续集成配置

```yaml
# .github/workflows/test.yml
name: CCCL Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点运行

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        cuda-version: ['11.8', '12.0', '12.1']
        build-type: ['Debug', 'Release']

    steps:
    - uses: actions/checkout@v3

    - name: Setup CUDA
      uses: Jimver/cuda-toolkit@v0.2.11
      with:
        cuda: ${{ matrix.cuda-version }}

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libgtest-dev libgmock-dev

    - name: Configure
      run: |
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=${{ matrix.build-type }} \
              -DCUDA_ARCHITECTURES="75;80;86" \
              ..

    - name: Build
      run: |
        cd build
        make -j$(nproc)

    - name: Run Tests
      run: |
        cd build
        ctest --verbose --output-on-failure

    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.cuda-version }}-${{ matrix.build-type }}
        path: build/Testing/
```

### 2. 测试报告生成

```cpp
// test_report_generator.cpp
#include "gpu_test_framework.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

class TestReportGenerator {
public:
    struct TestResult {
        std::string name;
        std::string status;  // "PASSED", "FAILED", "SKIPPED"
        double time_ms;
        std::string message;
        std::map<std::string, std::string> metadata;
    };

    void add_result(const TestResult& result) {
        results_.push_back(result);
    }

    void generate_json_report(const std::string& filename) {
        nlohmann::json report;

        report["test_suite"] = "CCCL GPU Tests";
        report["timestamp"] = get_current_timestamp();
        report["device_info"] = get_device_info();

        nlohmann::json test_results = nlohmann::json::array();
        for (const auto& result : results_) {
            nlohmann::json test_result;
            test_result["name"] = result.name;
            test_result["status"] = result.status;
            test_result["time_ms"] = result.time_ms;
            test_result["message"] = result.message;
            test_result["metadata"] = result.metadata;
            test_results.push_back(test_result);
        }

        report["tests"] = test_results;
        report["summary"] = generate_summary();

        std::ofstream file(filename);
        file << report.dump(4);
    }

    void generate_html_report(const std::string& filename) {
        std::ofstream file(filename);

        file << R"(<!DOCTYPE html>
<html>
<head>
    <title>CCCL GPU Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .test-item { margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }
        .passed { border-left-color: #4CAF50; }
        .failed { border-left-color: #f44336; }
        .skipped { border-left-color: #FF9800; }
        .summary { background-color: #e3f2fd; padding: 15px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>)";

        file << "<div class='header'>";
        file << "<h1>CCCL GPU Test Report</h1>";
        file << "<p>Generated: " << get_current_timestamp() << "</p>";
        file << "<p>Device: " << get_device_info()["name"] << "</p>";
        file << "</div>";

        auto summary = generate_summary();
        file << "<div class='summary'>";
        file << "<h2>Summary</h2>";
        file << "<p>Total: " << summary["total"] << "</p>";
        file << "<p>Passed: " << summary["passed"] << "</p>";
        file << "<p>Failed: " << summary["failed"] << "</p>";
        file << "<p>Skipped: " << summary["skipped"] << "</p>";
        file << "</div>";

        file << "<h2>Test Results</h2>";
        file << "<table>";
        file << "<tr><th>Test Name</th><th>Status</th><th>Time (ms)</th><th>Message</th></tr>";

        for (const auto& result : results_) {
            file << "<tr class='" << result.status << "'>";
            file << "<td>" << result.name << "</td>";
            file << "<td>" << result.status << "</td>";
            file << "<td>" << result.time_ms << "</td>";
            file << "<td>" << result.message << "</td>";
            file << "</tr>";
        }

        file << "</table>";
        file << "</body></html>";
    }

private:
    std::vector<TestResult> results_;

    std::string get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    nlohmann::json get_device_info() {
        int device_count;
        cudaGetDeviceCount(&device_count);

        nlohmann::json info;
        info["device_count"] = device_count;

        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            info["name"] = prop.name;
            info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
            info["total_memory_mb"] = prop.totalGlobalMem / (1024 * 1024);
        }

        return info;
    }

    nlohmann::json generate_summary() {
        nlohmann::json summary;
        summary["total"] = results_.size();
        summary["passed"] = 0;
        summary["failed"] = 0;
        summary["skipped"] = 0;

        for (const auto& result : results_) {
            if (result.status == "PASSED") summary["passed"]++;
            else if (result.status == "FAILED") summary["failed"]++;
            else if (result.status == "SKIPPED") summary["skipped"]++;
        }

        return summary;
    }
};
```

## 总结

通过构建全面的测试框架，我们为CCCL应用提供了：

**测试覆盖：**
1. **单元测试** - 验证单个组件的正确性
2. **集成测试** - 确保组件间的协作
3. **性能测试** - 保证性能不退化
4. **压力测试** - 验证极限条件下的行为
5. **回归测试** - 防止引入新bug

**自动化流程：**
1. **持续集成** - 自动化测试执行
2. **报告生成** - 详细的测试报告
3. **性能监控** - 跟踪性能趋势
4. **错误检测** - 快速发现问题

**质量保证：**
1. **代码覆盖率** - 确保充分测试
2. **基准对比** - 性能回归检测
3. **内存检查** - 内存安全验证
4. **多平台测试** - 跨环境兼容性

这个测试框架确保了我们开发的GPU数据流处理框架的质量和可靠性，为生产环境的部署提供了坚实的基础。

---

**系列总结：** 我们已经完成了CCCL的全面研究，从架构分析到功能扩展，再到测试框架。这套完整的技术方案展示了如何在GPU编程中充分利用CCCL的强大能力。

*本文基于CCCL 3.x版本，具体实现可能随版本更新而变化。*
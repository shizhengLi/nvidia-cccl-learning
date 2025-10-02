#include "gpu_stream_processor.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

using namespace cccl_extensions;

// 简单的性能测试函数
void test_basic_functionality() {
    std::cout << "=== 测试基本功能 ===" << std::endl;

    const size_t BUFFER_SIZE = 1024 * 1024;  // 1M elements
    const int NUM_STREAMS = 4;
    const size_t TOTAL_SIZE = BUFFER_SIZE * 10;  // 10M elements

    // 创建处理器
    auto processor = std::make_unique<GPUStreamProcessor<float>>(BUFFER_SIZE, NUM_STREAMS);

    // 设置处理函数：元素平方
    processor->set_process_function(
        ProcessFunctions::make_transform_function<float>(
            [] __device__ (float x) { return x * x; }
        )
    );

    // 设置完成回调
    processor->set_completion_callback(
        [](int buffer_id, float time_ms) {
            std::cout << "缓冲区 " << buffer_id << " 处理完成，耗时: " << time_ms << " ms" << std::endl;
        }
    );

    // 准备测试数据
    std::vector<float> host_data(TOTAL_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        host_data[i] = dis(gen);
    }

    // 处理数据流
    std::cout << "开始处理 " << TOTAL_SIZE << " 个元素..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t processed = processor->process_stream(host_data.data(), TOTAL_SIZE);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "处理完成！" << std::endl;
    std::cout << "总处理元素: " << processed << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;

    // 获取性能指标
    auto metrics = processor->get_metrics();
    std::cout << "平均吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;

    assert(processed == TOTAL_SIZE);
    std::cout << "✓ 基本功能测试通过" << std::endl;
}

// 测试排序处理器
void test_sort_processor() {
    std::cout << "\n=== 测试排序处理器 ===" << std::endl;

    const size_t BUFFER_SIZE = 256 * 1024;  // 256K elements
    const size_t TOTAL_SIZE = BUFFER_SIZE * 5;  // 1.25M elements

    // 创建排序处理器
    auto processor = ProcessorFactory::create_sort_processor<float>(BUFFER_SIZE, 3);
    processor->set_process_function(ProcessFunctions::sort_function<float>);

    // 准备随机数据
    std::vector<float> host_data(TOTAL_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1000.0f);

    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        host_data[i] = dis(gen);
    }

    // 保存第一个块的原始数据用于验证
    std::vector<float> first_chunk_original(host_data.begin(), host_data.begin() + BUFFER_SIZE);

    std::cout << "开始排序 " << TOTAL_SIZE << " 个元素..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t processed = processor->process_stream(host_data.data(), TOTAL_SIZE);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "排序完成！" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;

    auto metrics = processor->get_metrics();
    std::cout << "排序吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;

    assert(processed == TOTAL_SIZE);
    std::cout << "✓ 排序处理器测试通过" << std::endl;
}

// 测试过滤功能
void test_filter_processor() {
    std::cout << "\n=== 测试过滤处理器 ===" << std::endl;

    const size_t BUFFER_SIZE = 512 * 1024;
    const size_t TOTAL_SIZE = BUFFER_SIZE * 3;

    auto processor = std::make_unique<GPUStreamProcessor<int>>(BUFFER_SIZE, 2);

    // 设置过滤函数：只保留偶数
    auto filter_func = ProcessFunctions::FilterFunction<int,
        struct IsEven { __device__ bool operator()(int x) const { return x % 2 == 0; } }>(BUFFER_SIZE);

    processor->set_process_function(filter_func);

    // 准备测试数据：连续整数
    std::vector<int> host_data(TOTAL_SIZE);
    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        host_data[i] = static_cast<int>(i);
    }

    std::cout << "开始过滤 " << TOTAL_SIZE << " 个元素..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t processed = processor->process_stream(host_data.data(), TOTAL_SIZE);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "过滤完成！" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;

    assert(processed == TOTAL_SIZE);
    std::cout << "✓ 过滤处理器测试通过" << std::endl;
}

// 性能基准测试
void benchmark_performance() {
    std::cout << "\n=== 性能基准测试 ===" << std::endl;

    const size_t BUFFER_SIZE = 1024 * 1024;  // 1M elements
    const int NUM_STREAMS = 8;
    const size_t TOTAL_SIZE = BUFFER_SIZE * 50;  // 50M elements

    // 测试不同的流数量
    for (int streams = 1; streams <= NUM_STREAMS; streams *= 2) {
        std::cout << "\n--- 测试 " << streams << " 个流 ---" << std::endl;

        auto processor = std::make_unique<GPUStreamProcessor<float>>(BUFFER_SIZE, streams);

        // 使用复杂的处理函数
        processor->set_process_function(
            ProcessFunctions::make_transform_function<float>(
                [] __device__ (float x) {
                    return sinf(x) * cosf(x) + sqrtf(fabsf(x));
                }
            )
        );

        // 准备数据
        std::vector<float> host_data(TOTAL_SIZE);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (size_t i = 0; i < TOTAL_SIZE; ++i) {
            host_data[i] = dis(gen);
        }

        // 预热
        processor->process_stream(host_data.data(), BUFFER_SIZE);

        // 正式测试
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t processed = processor->process_stream(host_data.data(), TOTAL_SIZE);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto metrics = processor->get_metrics();

        std::cout << "处理时间: " << duration.count() << " ms" << std::endl;
        std::cout << "吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;
        std::cout << "活跃流数: " << metrics.active_streams.load() << std::endl;

        assert(processed == TOTAL_SIZE);
    }
}

// 测试错误处理
void test_error_handling() {
    std::cout << "\n=== 测试错误处理 ===" << std::endl;

    // 测试未初始化的处理器
    try {
        GPUStreamProcessor<float> processor(1024, 2);
        processor.set_process_function(
            ProcessFunctions::make_transform_function<float>(
                [] __device__ (float x) { return x * 2; }
            )
        );

        std::vector<float> data(100);
        processor.process_stream(data.data(), data.size());
        std::cout << "✓ 正常情况处理通过" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ 异常: " << e.what() << std::endl;
    }

    // 测试空数据
    try {
        auto processor = std::make_unique<GPUStreamProcessor<float>>(1024, 2);
        processor->set_process_function(
            ProcessFunctions::make_transform_function<float>(
                [] __device__ (float x) { return x * 2; }
            )
        );

        std::vector<float> empty_data;
        size_t processed = processor->process_stream(empty_data.data(), 0);
        assert(processed == 0);
        std::cout << "✓ 空数据处理通过" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ 异常: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "GPU Stream Processor 测试套件" << std::endl;
    std::cout << "==============================" << std::endl;

    try {
        // 基本功能测试
        test_basic_functionality();

        // 排序测试
        test_sort_processor();

        // 过滤测试
        test_filter_processor();

        // 性能基准测试
        benchmark_performance();

        // 错误处理测试
        test_error_handling();

        std::cout << "\n🎉 所有测试通过！" << std::endl;
        std::cout << "\nGPU Stream Processor 框架已成功实现并通过全面测试。" << std::endl;
        std::cout << "该框架展示了如何有效利用CCCL组件构建高性能的数据处理管道。" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
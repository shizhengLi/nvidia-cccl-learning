#include "gpu_stream_processor.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace cccl_extensions;

/**
 * @brief GPU Stream Processor 使用示例
 *
 * 这个示例展示了如何使用我们基于CCCL开发的GPU数据流处理框架
 * 来实现高性能的并行数据处理。
 */

void example_basic_usage() {
    std::cout << "=== 基本使用示例 ===" << std::endl;

    // 1. 创建处理器
    const size_t BUFFER_SIZE = 256 * 1024;  // 256K元素
    const int NUM_STREAMS = 4;

    auto processor = std::make_unique<GPUStreamProcessor<float>>(BUFFER_SIZE, NUM_STREAMS);

    // 2. 设置处理函数：简单的数学变换
    processor->set_process_function(
        ProcessFunctions::make_transform_function<float>(
            [] __device__ (float x) {
                return sinf(x) * cosf(x) + 1.0f;
            }
        )
    );

    // 3. 设置完成回调
    processor->set_completion_callback(
        [](int buffer_id, float time_ms) {
            std::cout << "  缓冲区 " << buffer_id << " 完成，耗时: "
                      << time_ms << " ms" << std::endl;
        }
    );

    // 4. 准备数据
    std::vector<float> data(BUFFER_SIZE * 6);  // 6个缓冲区的数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-3.14f, 3.14f);

    for (auto& val : data) {
        val = dis(gen);
    }

    // 5. 处理数据流
    std::cout << "开始处理数据流..." << std::endl;
    size_t processed = processor->process_stream(data.data(), data.size());

    // 6. 查看性能指标
    auto metrics = processor->get_metrics();
    std::cout << "处理完成！" << std::endl;
    std::cout << "总处理元素: " << processed << std::endl;
    std::cout << "平均吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;
}

void example_factory_usage() {
    std::cout << "\n=== 工厂函数使用示例 ===" << std::endl;

    // 使用工厂函数创建排序处理器
    auto sort_processor = ProcessorFactory::create_sort_processor<int>(128 * 1024, 3);

    // 准备随机数据
    std::vector<int> data(128 * 1024 * 4);  // 512K个整数
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000000);

    for (auto& val : data) {
        val = dis(gen);
    }

    std::cout << "开始排序 " << data.size() << " 个整数..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    size_t processed = sort_processor->process_stream(data.data(), data.size());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "排序完成！耗时: " << duration.count() << " ms" << std::endl;

    auto metrics = sort_processor->get_metrics();
    std::cout << "排序吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;
}

void example_custom_processing() {
    std::cout << "\n=== 自定义处理示例 ===" << std::endl;

    // 创建处理器
    auto processor = std::make_unique<GPUStreamProcessor<double>>(64 * 1024, 2);

    // 自定义复杂处理函数：统计计算
    processor->set_process_function([] (double* data, size_t size, cudaStream_t stream) {
        // 计算统计信息：均值和方差
        thrust::device_vector<double> d_data(data, data + size);

        // 计算均值
        double sum = thrust::reduce(thrust::cuda::par.on(stream),
                                  d_data.begin(), d_data.end(), 0.0);
        double mean = sum / size;

        // 计算方差
        thrust::device_vector<double> d_squared(size);
        thrust::transform(thrust::cuda::par.on(stream),
                         d_data.begin(), d_data.end(),
                         d_squared.begin(),
                         [mean] __device__ (double x) {
                             double diff = x - mean;
                             return diff * diff;
                         });

        double sum_sq = thrust::reduce(thrust::cuda::par.on(stream),
                                      d_squared.begin(), d_squared.end(), 0.0);
        double variance = sum_sq / size;

        // 将结果存储到原数组的前两个元素
        cudaMemcpyAsync(data, &mean, sizeof(double), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(data + 1, &variance, sizeof(double), cudaMemcpyDeviceToDevice, stream);
    });

    // 准备正态分布数据
    std::vector<double> normal_data(64 * 1024);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> ndist(100.0, 15.0);

    for (auto& val : normal_data) {
        val = ndist(gen);
    }

    std::cout << "开始统计计算..." << std::endl;
    size_t processed = processor->process_stream(normal_data.data(), normal_data.size());

    std::cout << "统计计算完成！" << std::endl;
    std::cout << "处理元素: " << processed << std::endl;

    // 获取性能指标
    auto metrics = processor->get_metrics();
    std::cout << "计算吞吐量: " << metrics.get_average_throughput_mbps() << " MB/s" << std::endl;
}

void example_pipeline_processing() {
    std::cout << "\n=== 流水线处理示例 ===" << std::endl;

    const size_t DATA_SIZE = 1024 * 1024;  // 1M元素

    // 第一阶段：数据生成和初步处理
    auto stage1 = ProcessorFactory::create_transform_processor<float>(256 * 1024, 3);
    stage1->set_process_function(
        ProcessFunctions::make_transform_function<float>(
            [] __device__ (float x) { return x * 2.0f + 1.0f; }
        )
    );

    // 第二阶段：过滤处理
    auto stage2 = std::make_unique<GPUStreamProcessor<float>>(256 * 1024, 2);
    stage2->set_process_function(
        ProcessFunctions::make_transform_function<float>(
            [] __device__ (float x) { return (x > 100.0f) ? x : 0.0f; }
        )
    );

    // 准备输入数据
    std::vector<float> input_data(DATA_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    for (auto& val : input_data) {
        val = dis(gen);
    }

    std::cout << "开始流水线处理..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 阶段1处理
    size_t processed1 = stage1->process_stream(input_data.data(), input_data.size());
    auto metrics1 = stage1->get_metrics();

    // 阶段2处理
    size_t processed2 = stage2->process_stream(input_data.data(), input_data.size());
    auto metrics2 = stage2->get_metrics();

    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "流水线处理完成！" << std::endl;
    std::cout << "总耗时: " << total_time.count() << " ms" << std::endl;
    std::cout << "阶段1吞吐量: " << metrics1.get_average_throughput_mbps() << " MB/s" << std::endl;
    std::cout << "阶段2吞吐量: " << metrics2.get_average_throughput_mbps() << " MB/s" << std::endl;
}

int main() {
    std::cout << "GPU Stream Processor 使用示例" << std::endl;
    std::cout << "===============================" << std::endl;

    try {
        // 基本使用
        example_basic_usage();

        // 工厂函数使用
        example_factory_usage();

        // 自定义处理
        example_custom_processing();

        // 流水线处理
        example_pipeline_processing();

        std::cout << "\n🎉 所有示例执行成功！" << std::endl;
        std::cout << "\n这个GPU数据流处理框架展示了如何：" << std::endl;
        std::cout << "• 利用CCCL组件构建高性能处理管道" << std::endl;
        std::cout << "• 实现异步并行处理" << std::endl;
        std::cout << "• 提供灵活的自定义处理函数" << std::endl;
        std::cout << "• 支持流式数据处理" << std::endl;
        std::cout << "• 实现性能监控和自动调优" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ 示例执行失败: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
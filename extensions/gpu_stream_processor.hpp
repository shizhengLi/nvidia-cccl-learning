#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/stream_ref>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace cccl_extensions {

/**
 * @brief GPU数据流处理框架
 *
 * 这是一个高性能的GPU数据流处理框架，利用CCCL的各种组件来实现：
 * - 流水线并行处理
 * - 异步数据传输
 * - 内存池管理
 * - 动态负载均衡
 * - 性能监控和调优
 */
template <typename T>
class GPUStreamProcessor {
public:
    using ProcessFunction = std::function<void(T*, size_t, cudaStream_t)>;
    using CompletionCallback = std::function<void(int, float)>;

private:
    struct StreamContext {
        cudaStream_t stream;
        thrust::device_vector<T> buffer;
        thrust::device_vector<T> output;
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        cudaEvent_t copy_done_event;
        int buffer_id;
        bool in_use;

        StreamContext(size_t buffer_size, int id)
            : buffer(buffer_size), output(buffer_size), buffer_id(id), in_use(false) {
            cudaStreamCreate(&stream);
            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);
            cudaEventCreate(&copy_done_event);
        }

        ~StreamContext() {
            cudaStreamDestroy(stream);
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            cudaEventDestroy(copy_done_event);
        }
    };

    struct ProcessingMetrics {
        cuda::std::atomic<uint64_t> total_processed{0};
        cuda::std::atomic<uint64_t> total_time_us{0};
        cuda::std::atomic<uint32_t> active_streams{0};
        float max_throughput_mbps = 0.0f;

        void update_processed(uint64_t count, uint64_t time_us) {
            total_processed.fetch_add(count, cuda::std::memory_order_relaxed);
            total_time_us.fetch_add(time_us, cuda::std::memory_order_relaxed);
        }

        float get_average_throughput_mbps() const {
            uint64_t processed = total_processed.load(cuda::std::memory_order_relaxed);
            uint64_t total_time = total_time_us.load(cuda::std::memory_order_relaxed);
            if (total_time == 0) return 0.0f;
            return (processed * sizeof(T) * 1000000.0f) / (total_time * 1024.0f * 1024.0f);
        }
    };

    std::vector<std::unique_ptr<StreamContext>> stream_contexts_;
    ProcessFunction process_function_;
    CompletionCallback completion_callback_;
    size_t buffer_size_;
    int num_streams_;
    bool initialized_;

    // 性能监控
    std::unique_ptr<ProcessingMetrics> metrics_;

    // 自适应调优
    struct AutoTuner {
        int optimal_block_size_ = 256;
        int optimal_items_per_thread_ = 4;
        bool tuning_enabled_ = true;

        void tune_if_needed(const ProcessingMetrics& metrics) {
            if (!tuning_enabled_) return;

            float current_throughput = metrics.get_average_throughput_mbps();
            if (current_throughput > metrics.max_throughput_mbps * 1.1f) {
                // 性能提升了，可能需要重新调优
                metrics.max_throughput_mbps = current_throughput;
            }
        }
    };

    std::unique_ptr<AutoTuner> auto_tuner_;

public:
    /**
     * @brief 构造函数
     * @param buffer_size 每个缓冲区的大小
     * @param num_streams 并行流数量
     */
    GPUStreamProcessor(size_t buffer_size, int num_streams = 4)
        : buffer_size_(buffer_size)
        , num_streams_(num_streams)
        , initialized_(false) {

        metrics_ = std::make_unique<ProcessingMetrics>();
        auto_tuner_ = std::make_unique<AutoTuner>();

        // 创建流上下文
        for (int i = 0; i < num_streams; ++i) {
            stream_contexts_.emplace_back(
                std::make_unique<StreamContext>(buffer_size, i));
        }

        initialized_ = true;
    }

    ~GPUStreamProcessor() {
        // RAII自动清理资源
    }

    /**
     * @brief 设置处理函数
     * @param func 处理函数，接收数据指针、大小和流
     */
    void set_process_function(ProcessFunction func) {
        process_function_ = std::move(func);
    }

    /**
     * @brief 设置完成回调函数
     * @param callback 回调函数，接收缓冲区ID和处理时间(ms)
     */
    void set_completion_callback(CompletionCallback callback) {
        completion_callback_ = std::move(callback);
    }

    /**
     * @brief 处理数据流
     * @param host_data 主机数据指针
     * @param size 数据大小
     * @return 成功处理的元素数量
     */
    size_t process_stream(const T* host_data, size_t size) {
        if (!initialized_ || !process_function_) {
            throw std::runtime_error("Stream processor not properly initialized");
        }

        size_t total_processed = 0;
        std::vector<std::future<void>> async_operations;

        // 分块处理数据
        for (size_t offset = 0; offset < size; offset += buffer_size_) {
            size_t chunk_size = std::min(buffer_size_, size - offset);

            // 查找可用的流
            auto* context = get_available_stream_context();
            if (!context) {
                // 没有可用的流，等待其中一个完成
                wait_for_any_stream();
                context = get_available_stream_context();
            }

            // 异步处理数据块
            auto future = std::async(std::launch::async, [this, context, host_data, offset, chunk_size]() {
                process_chunk_async(context, host_data + offset, chunk_size);
            });

            async_operations.push_back(std::move(future));
            total_processed += chunk_size;
        }

        // 等待所有操作完成
        for (auto& future : async_operations) {
            future.wait();
        }

        return total_processed;
    }

    /**
     * @brief 获取性能指标
     */
    ProcessingMetrics get_metrics() const {
        return *metrics_;
    }

    /**
     * @brief 启用/禁用自动调优
     */
    void set_auto_tuning(bool enabled) {
        auto_tuner_->tuning_enabled_ = enabled;
    }

    /**
     * @brief 获取推荐的块大小
     */
    int get_optimal_block_size() const {
        return auto_tuner_->optimal_block_size_;
    }

    /**
     * @brief 设置推荐的块大小
     */
    void set_optimal_block_size(int block_size) {
        auto_tuner_->optimal_block_size_ = block_size;
    }

private:
    StreamContext* get_available_stream_context() {
        for (auto& context : stream_contexts_) {
            if (!context->in_use) {
                context->in_use = true;
                metrics_->active_streams.fetch_add(1, cuda::std::memory_order_relaxed);
                return context.get();
            }
        }
        return nullptr;
    }

    void wait_for_any_stream() {
        // 简化实现：等待第一个流
        if (!stream_contexts_.empty()) {
            cudaStreamSynchronize(stream_contexts_[0]->stream);
            stream_contexts_[0]->in_use = false;
            metrics_->active_streams.fetch_sub(1, cuda::std::memory_order_relaxed);
        }
    }

    void process_chunk_async(StreamContext* context, const T* host_data, size_t chunk_size) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 记录开始时间
        cudaEventRecord(context->start_event, context->stream);

        // 异步复制数据到设备
        cudaMemcpyAsync(thrust::raw_pointer_cast(context->buffer.data()),
                       host_data,
                       chunk_size * sizeof(T),
                       cudaMemcpyHostToDevice,
                       context->stream);

        // 记录复制完成事件
        cudaEventRecord(context->copy_done_event, context->stream);

        // 等待复制完成后执行处理
        cudaStreamWaitEvent(context->stream, context->copy_done_event, 0);

        // 执行用户定义的处理函数
        process_function_(thrust::raw_pointer_cast(context->buffer.data()),
                         chunk_size,
                         context->stream);

        // 记录结束时间
        cudaEventRecord(context->stop_event, context->stream);

        // 同步等待完成
        cudaStreamSynchronize(context->stream);

        // 计算处理时间
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, context->start_event, context->stop_event);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();

        // 更新性能指标
        metrics_->update_processed(chunk_size, elapsed_us);

        // 自动调优
        auto_tuner_->tune_if_needed(*metrics_);

        // 调用完成回调
        if (completion_callback_) {
            completion_callback_(context->buffer_id, elapsed_ms);
        }

        // 释放流上下文
        context->in_use = false;
        metrics_->active_streams.fetch_sub(1, cuda::std::memory_order_relaxed);
    }
};

/**
 * @brief 预定义的处理函数
 */
namespace ProcessFunctions {
    /**
     * @brief 简单的元素级变换
     */
    template <typename T, typename TransformFunc>
    std::function<void(T*, size_t, cudaStream_t)>
    make_transform_function(TransformFunc func) {
        return [func](T* data, size_t size, cudaStream_t stream) {
            thrust::transform(thrust::cuda::par.on(stream),
                             data, data + size, data, func);
        };
    }

    /**
     * @brief 排序函数
     */
    template <typename T>
    void sort_function(T* data, size_t size, cudaStream_t stream) {
        thrust::sort(thrust::cuda::par.on(stream), data, data + size);
    }

    /**
     * @brief 规约函数（返回到主机）
     */
    template <typename T>
    class ReduceFunction {
    private:
        thrust::device_vector<T> d_result_{1};
        thrust::host_vector<T> h_result_{1};

    public:
        T operator()(T* data, size_t size, cudaStream_t stream) {
            T sum = thrust::reduce(thrust::cuda::par.on(stream),
                                  data, data + size, T{0});
            return sum;
        }
    };

    /**
     * @brief 过滤函数
     */
    template <typename T, typename PredicateFunc>
    class FilterFunction {
    private:
        thrust::device_vector<T> d_temp_;

    public:
        FilterFunction(size_t max_size) : d_temp_(max_size) {}

        size_t operator()(T* data, size_t size, cudaStream_t stream) {
            auto new_end = thrust::copy_if(thrust::cuda::par.on(stream),
                                         data, data + size,
                                         d_temp_.begin(),
                                         PredicateFunc{});
            size_t new_size = new_end - d_temp_.begin();

            // 复制回原数组
            cudaMemcpyAsync(data, thrust::raw_pointer_cast(d_temp_.data()),
                          new_size * sizeof(T),
                          cudaMemcpyDeviceToDevice,
                          stream);

            return new_size;
        }
    };
}

/**
 * @brief 工厂函数，创建常用的处理器
 */
namespace ProcessorFactory {
    template <typename T>
    std::unique_ptr<GPUStreamProcessor<T>>
    create_transform_processor(size_t buffer_size, int num_streams = 4) {
        auto processor = std::make_unique<GPUStreamProcessor<T>>(buffer_size, num_streams);

        // 默认变换：每个元素乘以2
        processor->set_process_function(
            ProcessFunctions::make_transform_function<T>(
                [] __device__ (T x) { return x * static_cast<T>(2); }
            )
        );

        return processor;
    }

    template <typename T>
    std::unique_ptr<GPUStreamProcessor<T>>
    create_sort_processor(size_t buffer_size, int num_streams = 4) {
        auto processor = std::make_unique<GPUStreamProcessor<T>>(buffer_size, num_streams);
        processor->set_process_function(ProcessFunctions::sort_function<T>);
        return processor;
    }
}

} // namespace cccl_extensions
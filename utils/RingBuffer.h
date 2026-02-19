#ifndef ORT_ARGUS_UTILS_RINGBUFFER_H
#define ORT_ARGUS_UTILS_RINGBUFFER_H

#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>

template <typename BufferType>
class RingBuffer
{
    public:
        template <typename... Args>
        RingBuffer(unsigned int number_of_buffers, Args&&... args);

        ~RingBuffer();

        /**
         * @brief fetch next available buffer
         */
        std::shared_ptr<BufferType> get_writable();

        /**
         * @brief fetch buffer to process the data
         */
        std::shared_ptr<BufferType> get_readable();

        /**
         * @brief number of buffers
         */
        std::size_t buffer_size() const;

        /**
         * @brief number of writable buffers
         */
        unsigned int number_of_writable_buffers() const;

        /**
         * @brief number of readable buffers
         */
        unsigned int number_of_readable_buffers() const;

        /**
         * @brief set abort
         */
        void abort(bool value);

        /**
         * @brief return the status
         */
        bool abort() const;

        /**
         * @brief deleter function for the shared pointer
         */
        void push_function(BufferType* ptr);

        /**
         * @brief function to write to buffer
         */
        template <typename DataType>
        int write_to_buffer(typename DataType::const_iterator begin_it, typename DataType::const_iterator end_it);

        void release_buffer(std::shared_ptr<BufferType>& temp);

    private:
        std::function<void(BufferType*)> _buffer_deleter; // deleter function for ringbuffer
        unsigned _number_of_modules;
        unsigned _number_of_samples;
        size_t _buffer_size;
        size_t _write_location;
        size_t _number_of_buffers;
        std::atomic<size_t> _current_buffer_index;
        std::deque<std::shared_ptr<BufferType>> _readable_queue;
        std::deque<std::shared_ptr<BufferType>> _writeable_queue;
        std::mutex _read_mutex;
        std::mutex _write_mutex;
        volatile bool _abort;
        volatile bool _read_flag;
        volatile bool _write_flag;
};

#include "detail/RingBuffer.cpp"

#endif //ORT_ARGUS_UTILS_RINGBUFFER_H
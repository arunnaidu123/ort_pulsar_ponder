#ifndef SKA_PANDA_PACKETSTREAMLITE_RINGBUFFER_H
#define SKA_PANDA_PACKETSTREAMLITE_RINGBUFFER_H

#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>
#include "argus/utils/AlignedAllocator.h"
template <typename NumericalRep>
class RcptRingBuffer
{

    public:
        typedef typename std::pair<unsigned, std::shared_ptr<std::vector<NumericalRep>>> BufferType;

    public:
        RcptRingBuffer(unsigned int payloads_per_buffer, unsigned int payload_size, unsigned int number_of_buffers);
        ~RcptRingBuffer();

        /**
         * @brief fetch buffer to write packets
         */
        std::shared_ptr<BufferType>& get_writable(unsigned long sequence_number);

        /**
         * @brief fetch buffer to read packets
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
         * @brief return the payload size
         */
        unsigned int payload_size() const;

        /**
         * @brief return number of payloads per buffer
         */
        unsigned int payloads_per_buffer() const;

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


    private:
        std::function<void(BufferType*)> _buffer_deleter; // deleter function for ringbuffer
        unsigned int _payloads_per_buffer; // number of payloads expected to be in a given buffer
        unsigned int _payload_size; // size of each payload
        unsigned _number_of_buffers;
        unsigned long _start_packet_sequence_number;
        std::deque<std::shared_ptr<BufferType>> _readable_queue;
        std::deque<std::shared_ptr<BufferType>> _writeable_queue;
        std::mutex _read_mutex;
        std::mutex _write_mutex;
        volatile bool _abort;
        volatile bool _read_flag;
        volatile bool _write_flag;
};

#include "detail/RcptRingBuffer.cpp"

#endif
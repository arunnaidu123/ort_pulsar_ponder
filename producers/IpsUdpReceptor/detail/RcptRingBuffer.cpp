#include <tuple>
#include <numeric>
#include <iostream>
#include <stdexcept>

template <typename NumericalRep>
RcptRingBuffer<NumericalRep>::RcptRingBuffer(unsigned int payloads_per_buffer, unsigned int payload_size, unsigned int number_of_buffers)
    : _payloads_per_buffer(payloads_per_buffer)
    , _payload_size(payload_size)
    , _number_of_buffers(number_of_buffers)
    , _start_packet_sequence_number(0)
    ,_abort(false)
{
    std::lock_guard<std::mutex> lock(_write_mutex);
    _buffer_deleter = std::bind(&RcptRingBuffer::push_function, this, std::placeholders::_1);

    for (unsigned int i = 0; i < number_of_buffers; ++i) {
        BufferType* raw = new BufferType(
            std::make_pair(0, std::make_shared<VoltageSeries<NumericalRep>>(1, _payloads_per_buffer * _payload_size))
        );

        _writeable_queue.push_back(std::shared_ptr<BufferType>(raw, _buffer_deleter));
    }
}

template <typename NumericalRep>
RcptRingBuffer<NumericalRep>::~RcptRingBuffer()
{
}

template <typename NumericalRep>
void RcptRingBuffer<NumericalRep>::push_function(BufferType* ptr)
{
    if (!_abort)
    {
        //std::cout << "pushing data " << (void*)ptr<<"\n";
        try
        {
            std::lock_guard<std::mutex> lock(_write_mutex);
            ptr->first = 0;
            std::fill(ptr->second->begin(), ptr->second->end(), 0);
            ptr->second->mjd_day(0);


            _writeable_queue.push_back(std::shared_ptr<BufferType>(ptr, _buffer_deleter));
        }
        catch (...)
        {
            delete ptr;
        }
    }
    else
    {
        delete ptr;
    }
}

template <typename NumericalRep>
std::shared_ptr<typename RcptRingBuffer<NumericalRep>::BufferType>&
RcptRingBuffer<NumericalRep>::get_writable(unsigned long sequence_number)
{
    std::lock_guard<std::mutex> lock(_write_mutex);
    unsigned buffer_index = 0;
    if (sequence_number > _start_packet_sequence_number)
    {
        buffer_index =
            (sequence_number - _start_packet_sequence_number) / _payloads_per_buffer;


        if (buffer_index >= _writeable_queue.size())
        {
            std::cout << "Break in data stream. Exiting "<<buffer_index<<" "<<_writeable_queue.size()<<"\n";
            abort(true);
            throw std::runtime_error("Break in data stream");
        }

        if (buffer_index > _number_of_buffers/2)
        {
            {
                std::lock_guard<std::mutex> rlock(_read_mutex);
                _readable_queue.push_back(_writeable_queue.front());
            }
            _read_cv.notify_one();

            _writeable_queue.pop_front();
            _start_packet_sequence_number += _payloads_per_buffer;
            --buffer_index;
        }
    }


    _writeable_queue[buffer_index]->first += 1;
    //if(buffer_index==4) std::cout<<sequence_number<<" "<<_writeable_queue[buffer_index]->first<<" "<<_payloads_per_buffer<<"\n";
    return _writeable_queue[buffer_index];
}

template <typename NumericalRep>
std::shared_ptr<typename RcptRingBuffer<NumericalRep>::BufferType>
RcptRingBuffer<NumericalRep>::get_readable()
{
    std::unique_lock<std::mutex> lock(_read_mutex);
    _read_cv.wait(lock, [this]{ return !_readable_queue.empty() || abort(); });

    if (_readable_queue.empty())
        throw std::runtime_error("Unable to return a readable object");

    auto temp = _readable_queue.front();
    _readable_queue.pop_front();

    int missing_packets =
        int(_payloads_per_buffer) - int(temp->first);

    if (missing_packets)
        std::cout << "missing packets " << missing_packets << "\n";

    return temp;
}

template <typename NumericalRep>
std::size_t RcptRingBuffer<NumericalRep>::buffer_size() const
{
    return _payloads_per_buffer*_payload_size;
}

template <typename NumericalRep>
unsigned int RcptRingBuffer<NumericalRep>::number_of_writable_buffers() const
{
    return _writeable_queue.size();
}

template <typename NumericalRep>
unsigned int RcptRingBuffer<NumericalRep>::number_of_readable_buffers() const
{
    return _readable_queue.size();
}

template <typename NumericalRep>
unsigned int RcptRingBuffer<NumericalRep>::payload_size() const
{
    return _payload_size;
}

template <typename NumericalRep>
unsigned int RcptRingBuffer<NumericalRep>::payloads_per_buffer() const
{
    return _payloads_per_buffer;
}

template <typename NumericalRep>
void RcptRingBuffer<NumericalRep>::abort(bool value)
{
    _abort=value;
    _read_cv.notify_all();
}

template <typename NumericalRep>
bool RcptRingBuffer<NumericalRep>::abort() const
{
    return _abort;
}

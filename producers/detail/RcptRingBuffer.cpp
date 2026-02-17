#include <tuple>
#include <numeric>
#include <iostream>

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
            std::make_pair(0, std::make_shared<std::vector<NumericalRep>>(_payloads_per_buffer * _payload_size))
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
            ptr->first = 0;
            std::fill(ptr->second->begin(), ptr->second->end(), 0);
            std::lock_guard<std::mutex> lock(_write_mutex);
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
            std::cout << "Break in data stream. Exiting\n";
            abort(true);
        }

        while (buffer_index > 2)
        {
            {
                std::lock_guard<std::mutex> rlock(_read_mutex);
                _readable_queue.push_back(_writeable_queue.front());
            }

            _writeable_queue.pop_front();
            _start_packet_sequence_number += _payloads_per_buffer;
            --buffer_index;
        }
    }

    _writeable_queue[buffer_index]->first += 1;

    return _writeable_queue[buffer_index];
}

//template <typename NumericalRep>
//std::shared_ptr<typename RcptRingBuffer<NumericalRep>::BufferType>& RcptRingBuffer<NumericalRep>::get_writable(unsigned long sequence_number)
//{
//    unsigned buffer_index = 0;
//    if(sequence_number>_start_packet_sequence_number)
//    {
//        buffer_index = (sequence_number-_start_packet_sequence_number)/_payloads_per_buffer;
//
//        if(buffer_index>_writeable_queue.size())
//        {
//            std::cout<<"Break in data stream. Exiting";
//            abort(true);
//        }
//
//        if(buffer_index>2)
//        {
//            std::lock_guard<std::mutex> lock(_read_mutex);
//            for(unsigned int i=1; i<buffer_index;++i)
//            {
//                _readable_queue.push_back(_writeable_queue.front());
//                _writeable_queue.pop_front();
//                _start_packet_sequence_number += _payloads_per_buffer;
//            }
//            buffer_index=2;
//        }
//    }
//    {
//        std::lock_guard<std::mutex> lock(_write_mutex);
//        (*_writeable_queue[buffer_index]).first += 1;
//    }
//    return _writeable_queue[buffer_index];
//}


template <typename NumericalRep>
std::shared_ptr<typename RcptRingBuffer<NumericalRep>::BufferType>
RcptRingBuffer<NumericalRep>::get_readable()
{
    while (!abort())
    {
        std::lock_guard<std::mutex> lock(_read_mutex);

        if (!_readable_queue.empty())
        {
            auto temp = _readable_queue.front();
            _readable_queue.pop_front();

            int missing_packets =
                int(_payloads_per_buffer) - int(temp->first);

            if (missing_packets)
                std::cout << "missing packets " << missing_packets << "\n";

            return temp;
        }
    }

    throw std::runtime_error("Unable to return a readable object");
}

//template <typename NumericalRep>
//std::shared_ptr<typename RcptRingBuffer<NumericalRep>::BufferType> RcptRingBuffer<NumericalRep>::get_readable()
//{
//    BufferType temp;
//    bool flag=true;
//
//    while(flag && !abort())
//    {
//        {
//            std::lock_guard<std::mutex> lock(_read_mutex);
//            flag = _readable_queue.empty();
//        }
//
//        if(!flag)
//        {
//            std::lock_guard<std::mutex> lock(_read_mutex);
//            auto temp = _readable_queue.front();
//            _readable_queue.pop_front();
//            int missing_packets = ((int)_payloads_per_buffer)-((int)(*temp).first);
//            if(missing_packets)
//            {
//                std::cout<<"missing packets "<<missing_packets<<"\n";
//            }
//            return temp;
//        }
//    }
//
//    throw std::runtime_error("Unable to return a readable object\n");
//}

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
}

template <typename NumericalRep>
bool RcptRingBuffer<NumericalRep>::abort() const
{
    return _abort;
}
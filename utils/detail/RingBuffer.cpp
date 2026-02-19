//#include "RingBuffer.h"
#include <tuple>
#include <numeric>
#include <iostream>

template <typename BufferType>
template <typename... Args>
RingBuffer<BufferType>::RingBuffer(unsigned int number_of_buffers, Args&&... args)
    : _buffer_size(0)
    , _write_location(0)
    , _number_of_buffers(number_of_buffers)
    , _current_buffer_index(0)
    , _abort(false)
{
    std::lock_guard<std::mutex> lock(_write_mutex);
    _buffer_deleter = std::bind(&RingBuffer::push_function, this, std::placeholders::_1);

    for (unsigned int i = 0; i < _number_of_buffers; ++i) {
        BufferType* raw = new BufferType(args...);
        if(_buffer_size==0) _buffer_size = raw->size();
        _writeable_queue.push_back(std::shared_ptr<BufferType>(raw, _buffer_deleter));
    }
}

template <typename BufferType>
RingBuffer<BufferType>::~RingBuffer()
{
}

template <typename BufferType>
void RingBuffer<BufferType>::push_function(BufferType* ptr)
{
    if (!_abort)
    {
        //std::cout << "pushing data " << (void*)ptr<<"\n";
        try
        {
            std::fill(ptr->begin(), ptr->end(), 0);
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

template <typename BufferType>
std::shared_ptr<BufferType> RingBuffer<BufferType>::get_writable()
{
    std::lock_guard<std::mutex> lock(_write_mutex);

    if (_writeable_queue.empty())
        throw std::runtime_error("No writable buffers available");

    auto buffer = _writeable_queue.front();
    _writeable_queue.pop_front();
    return buffer;   // safe copy of shared_ptr
}

//template <typename BufferType>
//std::shared_ptr<BufferType>& RingBuffer<BufferType>::get_writable()
//{
//    if(_current_buffer_index>1)
//    {
//        std::lock_guard<std::mutex> lock(_read_mutex);
//        while(_current_buffer_index!=0)
//        {
//            _readable_queue.push_back(_writeable_queue.front());
//            _writeable_queue.pop_front();
//            _current_buffer_index--;
//        }
//    }
//    return _writeable_queue[_current_buffer_index];
//}

//template <typename BufferType>
//template <typename DataType>
//int RingBuffer<BufferType>::write_to_buffer(typename DataType::const_iterator begin_it, typename DataType::const_iterator end_it)
//{
//    auto buffer = this->get_writable();
//
//    const size_t number_of_elements = std::distance(begin_it, end_it);
//
//    // How much space until end of buffer
//    const size_t space_to_end = _buffer_size - _write_location;
//    if (number_of_elements <= space_to_end)
//    {
//        // Case 1: single contiguous copy
//        std::copy(begin_it, end_it, buffer->begin() + _write_location);
//        _write_location += number_of_elements;
//
//        if(_write_location == _buffer_size)
//        {
//            _current_buffer_index++;
//            std::lock_guard<std::mutex> lock(_read_mutex);
//            std::lock_guard<std::mutex> lock1(_write_mutex);
//            while(_current_buffer_index>=1)
//            {
//                _readable_queue.push_back(_writeable_queue.front());
//                _writeable_queue.pop_front();
//                _current_buffer_index--;
//            }
//            _write_location = 0;
//        }
//
//    }
//    else
//    {
//        // Case 2: wrap-around copy
//
//        // First chunk: write till end of buffer
//        auto mid_it = begin_it + space_to_end;
//        std::copy(begin_it, mid_it, buffer->begin() + _write_location);
//
//        _current_buffer_index++;
//        _write_location = 0;
//
//        buffer = this->get_writable();
//        // Second chunk: wrap to beginning
//        std::copy(mid_it, end_it, buffer->begin());
//        _write_location = number_of_elements - space_to_end;
//    }
//
//    return 0;
//}

template <typename BufferType>
void RingBuffer<BufferType>::release_buffer(std::shared_ptr<BufferType>& temp)
{
        std::lock_guard<std::mutex> lock1(_read_mutex);
        _readable_queue.push_back(temp);
}

template <typename BufferType>
std::shared_ptr<BufferType> RingBuffer<BufferType>::get_readable()
{
    while (!abort())
    {
        std::lock_guard<std::mutex> lock(_read_mutex);

        if (!_readable_queue.empty())
        {
            auto buffer = _readable_queue.front();
            _readable_queue.pop_front();
            return buffer;   // return by value
        }
    }

    throw std::runtime_error("Unable to return a readable object");
}

template <typename BufferType>
std::size_t RingBuffer<BufferType>::buffer_size() const
{
    return _buffer_size;
}

template <typename BufferType>
unsigned int RingBuffer<BufferType>::number_of_writable_buffers() const
{
    return _writeable_queue.size();
}

template <typename BufferType>
unsigned int RingBuffer<BufferType>::number_of_readable_buffers() const
{
    return _readable_queue.size();
}

template <typename BufferType>
void RingBuffer<BufferType>::abort(bool value)
{
    _abort=value;
}

template <typename BufferType>
bool RingBuffer<BufferType>::abort() const
{
    return _abort;
}
#ifndef ORT_OWFA_ARGUS_UTILS_ALIGNEDALLOCATOR_H
#define ORT_OWFA_ARGUS_UTILS_ALIGNEDALLOCATOR_H

#include <cstdlib>
#include <memory>
#include <new>
#include <cstddef>

template <typename T, std::size_t Alignment>
struct AlignedAllocator
{
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        free(p);
    }
};

#endif //ORT_OWFA_ARGUS_UTILS_ALIGNEDALLOCATOR_H
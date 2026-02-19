#ifndef ORT_OWFA_ARGUS_UTILS_MJDTIME_H
#define ORT_OWFA_ARGUS_UTILS_MJDTIME_H


#include <cstdint>
#include <stdexcept>

class MJDTime
{
public:
    uint32_t day;        // MJD day
    uint32_t seconds;    // 0 – 86399
    uint32_t nanoseconds; // 0 – 999,999,999

    MJDTime(uint32_t d = 0,
            uint32_t s = 0,
            uint32_t ns = 0)
        : day(0), seconds(0), nanoseconds(0)
    {
        normalize();
    }

    MJDTime(MJDTime const& value)
        : day(value.day), seconds(value.seconds), nanoseconds(value.nanoseconds)
    {
        normalize();
    }

    void add(uint64_t add_seconds,
             uint64_t add_nanoseconds = 0)
    {
        nanoseconds += add_nanoseconds;
        seconds += add_seconds;

        normalize();
    }

private:
    void normalize()
    {
        // Fix nanoseconds overflow
        if (nanoseconds >= 1'000'000'000)
        {
            seconds += nanoseconds / 1'000'000'000;
            nanoseconds %= 1'000'000'000;
        }

        // Fix seconds overflow
        if (seconds >= 86400)
        {
            day += seconds / 86400;
            seconds %= 86400;
        }
    }
};

#endif //ORT_OWFA_ARGUS_UTILS_MJDTIME_H
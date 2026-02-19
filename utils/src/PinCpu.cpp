#include "PinCpu.h"
#include <sched.h>
#include <unistd.h>
#include <cstdio>

void pin_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);

    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0) {
        perror("sched_setaffinity");
    }
}
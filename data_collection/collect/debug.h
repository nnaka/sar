#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#if DEBUG
#define LOG(fmt, ...) \
    ((void)fprintf(stderr, "%s:%d:%s(): " fmt"...\n", __FILE__, \
            __LINE__, __func__, __VA_ARGS__));
#else
#define LOG(fmt, ...) ((void)0)
#endif

// Prints errno message if there is one, otherwise `LOG`s `msg`
inline void check_or_exit(bool cond, const char *msg) {
    if (cond) {
        (errno) ? perror(msg) : LOG("%s", msg);
        exit(EXIT_FAILURE);
    }
}

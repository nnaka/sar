#pragma once

#include <stdlib.h>
#include <stdio.h>

#define LOG(fmt, ...) \
    do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt"...\n", __FILE__, \
            __LINE__, __func__, __VA_ARGS__); } while (0)

inline void check_or_exit(bool cond, const char *msg) {
    if (cond) { perror(msg); exit(EXIT_FAILURE); }
}

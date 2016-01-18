#pragma once

#define LOG(fmt, ...) \
    do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt"...\n", __FILE__, \
            __LINE__, __func__, __VA_ARGS__); } while (0)

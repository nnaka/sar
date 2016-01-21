#pragma once

// pulson.h
// PulsOn collects data from a GPS over USB

#include <string>

class PulsOn {
    public:
        PulsOn(const std::string &);
        void collect();
};

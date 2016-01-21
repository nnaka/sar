#pragma once

// pulse_history.h
// PulseHistory collects synchronized data from a GPS and radar

#include <string>
#include <vector>

#include "piksi.h"
#include "pulson.h"

class PulseHistory {
    public:
        PulseHistory(const std::string &, const std::string &);
        void collect();
    private:
        Piksi gps;
        PulsOn radar;
        std::vector<std::string> pulse_history;

        const int pulsesPerLoc;
};

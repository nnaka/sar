#pragma once

// pulse_history.h
// PulseHistory collects synchronized data from a GPS and radar

#include <string>
#include <vector>
#include <stdlib.h>

#include "piksi.h"
#include "pulson.h"

class PulseHistory {
    public:
        PulseHistory(const std::string &, const std::string &);
        void collect();
        void clearHistory();
        friend std::ostream& operator<<(std::ostream& os,
                const PulseHistory& ph);
    private:
        Piksi gps;
        PulsOn radar;
        std::vector<std::string> pulseHistory;

        const int pulsesPerLoc;
};

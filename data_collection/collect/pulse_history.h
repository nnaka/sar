#pragma once

// pulse_history.h
// PulseHistory collects synchronized data from a GPS and radar

#include <string>

class PulseHistory {
    public:
        PulseHistory(const std::string &, const std::string &);

        void collect();
    private:
        // Piksi gps;
        // PulsOn radar;
        // vector<string> pulse_history;
};

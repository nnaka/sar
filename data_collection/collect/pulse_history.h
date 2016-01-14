#pragma once

// pulse_history.h
// PulseHistory collects synchronized data from a GPS and radar

class PulseHistory {
    public:
        PulseHistory();
        PulseHistory(const char *, const char *);

        void collect();
    private:
        // Piksi gps;
        // PulsOn radar;
        // vector<string> pulse_history;
};

#include "pulse_history.h"

using namespace std;

PulseHistory::PulseHistory(const string &gps_port, const string &radar_port) :
    gps(gps_port), radar(radar_port), pulsesPerLoc(1) {}

// Collects 1 GPS pulse for X number of radar pulses such that the radar pulses
// are approximately associated to that 1 GPS pulse in space and time.
//
// @raises CollectionError
void PulseHistory::collect() {
    gps.collect();

    for (int i = 0; i < pulsesPerLoc; ++i) {
        radar.collect();
    }
}

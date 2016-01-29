#include "pulse_history.h"
#include <iostream>

using namespace std;

PulseHistory::PulseHistory(const string &gpsPort, const string &radarPort) :
    gps(gpsPort), radar(radarPort), pulsesPerLoc(1) {}

// Collects 1 GPS pulse for X number of radar pulses such that the radar pulses
// are approximately associated to that 1 GPS pulse in space and time.
//
// @raises CollectionError
void PulseHistory::collect() {
    pulsonInfo info;

    msg_pos_llh_t  pos_info;
    msg_gps_time_t gps_info;

    gps.collect(pos_info, gps_info);

    for (int i = 0; i < pulsesPerLoc; ++i) {
        radar.collect(info);
    }
}

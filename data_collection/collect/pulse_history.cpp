#include "pulse_history.h"
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>

using namespace std;

PulseHistory::PulseHistory(const string &gpsPort, const string &radarPort) :
    gps(gpsPort), radar(radarPort), pulsesPerLoc(1) {}

// Collects 1 GPS pulse for (100?) number of radar pulses such that the radar pulses
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

    stringstream ss;

    for (unsigned int i = 0; i < info.msg.scanInfo.numSamplesTotal; i++) {
        ss << info.scan[i]  << ", ";
    }
    
    ss << pos_info.lat    <<   ", "
       << pos_info.lon    <<   ", "
       << pos_info.height <<   ", " << endl << endl;

    pulseHistory.push_back(ss.str());
}

// Clears `pulseHistory` array
void PulseHistory::clearHistory() {
    pulseHistory.clear();
}

ostream& operator<<(ostream& os, const PulseHistory& ph) {
    for (auto pulse : ph.pulseHistory) { os << pulse; }
    return os;
}

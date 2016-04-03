#include "pulse_history.h"
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

    cout << "starting scan: \n";

    for (int i = 0; i < info.msg.scanInfo.numSamplesTotal; i++) {
        cout << info.scan[i]  << ", ";
    }

    cout << "\n\n";
        /*
   cout << "\n\n ========== Radar Stand Alone Data ========\n\n"
         << "Radar data: "          << info.msg.scanInfo.scan                   << "; \n"
         << "NumSamples: "          << info.msg.scanInfo.numSamplesTotal        << "; \n"
         << "SourceID: "            << info.msg.scanInfo.sourceId               << "; \n"
         << "scanStartPS: "         << info.msg.scanInfo.scanStartPs            << "; \n"
         << "scanStopPS: "          << info.msg.scanInfo.scanStopPs             << "; \n"
         << "scanStepBins: "        << info.msg.scanInfo.scanStepBins           << "; \n"
         << "transmitGain: "        << 63                                       << "; \n"     // from txGain
         << "codeChannel: "         << 0                                        << "; \n"     // from codeChannel
         << "pii: "                 << 12                                       << "; \n";  // from base integration index
    
    cout << "\n ========== GPS Stand Alone Data ==========\n\n"
         << "Position x: "          << pos_info.lat                             << "; \n"
         << "Position y: "          << pos_info.lon                             << "; \n"
         << "Height: "              << pos_info.height                          << "; \n"
         << "GPS week: "            << gps_info.wn                              << "; \n"
         << "GPS time: "            << gps_info.tow                             << "; \n\n";
*/
}


// Radar Data;
// GPS Latitude;
// GPS Longitude;
// GPS Height;
// sourceID;
// scanStartPs;
// scanStopPs;
// scanStepBins;
// transmitGain ;
// antennaID - not used;
// codeChannel;
// pulseIntegrationIndex;
// operationMode - not used;
// scanIntervalTime_ms - not used

/*

 cout << "\n\n ========== Radar Stand Alone Data ========\n\n"
 << "Radar data: "          << info.msg.scanInfo.scan                   << "; \n"
 << "NumSamples: "          << info.msg.scanInfo.numSamplesTotal        << "; \n"
 << "SourceID: "            << info.msg.scanInfo.sourceId               << "; \n"
 << "scanStartPS: "         << info.msg.scanInfo.scanStartPs            << "; \n"
 << "scanStopPS: "          << info.msg.scanInfo.scanStopPs             << "; \n"
 << "scanStepBins: "        << info.msg.scanInfo.scanStepBins           << "; \n"
 << "transmitGain: "        << 63                                       << "; \n"     // from txGain
 << "codeChannel: "         << 0                                        << "; \n"     // from codeChannel
 << "pii: "                 << 12                                       << "; \n";  // from base integration index
 
 cout << "\n ========== GPS Stand Alone Data ==========\n\n"
 << "Position x: "          << pos_info.lat                             << "; \n"
 << "Position y: "          << pos_info.lon                             << "; \n"
 << "Height: "              << pos_info.height                          << "; \n"
 << "GPS week: "            << gps_info.wn                              << "; \n"
 << "GPS time: "            << gps_info.tow                             << "; \n\n";

*/

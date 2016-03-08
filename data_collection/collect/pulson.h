#pragma once

// pulson.h
// PulsOn collects data from a GPS over USB

#include <stdlib.h>
#include <string>

#include "mrmIf.h"
#include "mrm.h"

struct pulsonInfo : public mrmInfo {
    pulsonInfo()  { scan = nullptr; }
    ~pulsonInfo() { if (scan) { free(scan); } }
};

class PulsOn {
    public:
        PulsOn(const std::string &);
        ~PulsOn();

        void collect(pulsonInfo &);
    private:
        //
        // Configuration struct and defaults
        //

        mrmConfiguration config;

        const uint32_t DEFAULT_SCAN_START     = 17400;
        // DEFAULT_SCAN_STOP is computed
        const uint16_t DEFAULT_MAX_DISTANCE   = 13;
        const uint32_t DEFAULT_BASEII         = 12;
        const uint8_t  DEFAULT_TX_GAIN        = 63;
        const uint8_t  DEFAULT_CODE_CHANNEL   = 0;
	    const uint8_t  DEFAULT_ANTENNA_MODE   = 3;
	    const uint16_t DEFAULT_SCAN_RESOLUTION_BINS = 32;

        uint32_t userScanStart;
        uint32_t userScanStop;
	    uint16_t userBaseII;
        uint8_t  userTxGain;
        uint8_t  userCodeChannel;
        uint8_t  userAntennaMode;
        uint16_t userScanResolutionBins;

        //
        // Control parameters and defaults
        //

        const uint16_t DEFAULT_SCAN_COUNT    = 1;
        const uint32_t DEFAULT_SCAN_INTERVAL = 0;
        
        uint16_t userScanCount;
        uint32_t userScanInterval;

};

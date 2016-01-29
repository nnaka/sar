#pragma once

// pulson.h
// PulsOn collects data from a GPS over USB

#include <stdlib.h>
#include <string>

#include "mrmIf.h"
#include "mrm.h"

struct pulsonInfo : public mrmInfo {
    pulsonInfo()  { scan = NULL; }
    ~pulsonInfo() { if (scan) { free(scan); } }
};

class PulsOn {
    public:
        PulsOn(const std::string &);
        ~PulsOn();

        void collect(pulsonInfo &info);
    private:
        const int DEFAULT_BASEII        = 12,
                  DEFAULT_SCAN_START    = 10000,
                  DEFAULT_SCAN_STOP     = 39297,
                  DEFAULT_SCAN_COUNT    = 1,
                  DEFAULT_SCAN_INTERVAL = 125000,
                  DEFAULT_TX_GAIN       = 63;
        
        mrmConfiguration config;

	    int userBaseII,
            userScanStart,
            userScanStop,
            userScanCount,
            userScanInterval,
            userTxGain;
};

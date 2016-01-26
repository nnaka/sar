#pragma once

// pulson.h
// PulsOn collects data from a GPS over USB

#include <string>

#include "mrmIf.h"
#include "mrm.h"

class PulsOn {
    public:
        PulsOn(const std::string &);
        ~PulsOn();
        void collect();
    private:
        void processInfo(mrmInfo *info, FILE *fp, int printInfo);
        void mrmSampleExit(void);

        const int DEFAULT_BASEII        = 12,
                  DEFAULT_SCAN_START    = 10000,
                  DEFAULT_SCAN_STOP     = 39297,
                  DEFAULT_SCAN_COUNT    = 5,
                  DEFAULT_SCAN_INTERVAL = 125000,
                  DEFAULT_TX_GAIN       = 63;
        
        bool connected, userPrintInfo;

        mrmConfiguration config;

	    int userBaseII,
            userScanStart,
            userScanStop,
            userScanCount,
            userScanInterval,
            userTxGain;
};

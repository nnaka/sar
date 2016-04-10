#include "pulson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <time.h>

#include "debug.h"

using namespace std;

const auto C = 299792458; // speed of light

// NOTE: Most memory leaks marked below result from not calling
// `mrmSampleExit()` So `mrmIfClose()` is never called, and so the socket is
// forcibly released by the OS which can cause 'bind: Socket already in use'
// errors.

PulsOn::PulsOn(const string & radioAddr) :
    userScanStart(DEFAULT_SCAN_START),
    userScanStop(userScanStart + (2 * (float)DEFAULT_MAX_DISTANCE / C) * 1e12),
	userBaseII(DEFAULT_BASEII),
    userTxGain(DEFAULT_TX_GAIN),
    userCodeChannel(DEFAULT_CODE_CHANNEL),
    userAntennaMode(DEFAULT_ANTENNA_MODE),
    userScanResolutionBins(DEFAULT_SCAN_RESOLUTION_BINS),
    userScanCount(DEFAULT_SCAN_COUNT),
    userScanInterval(DEFAULT_SCAN_INTERVAL)
{
    int mode;

    // config and status strucs
    mrmMsg_GetStatusInfoConfirm statusInfo;

    // announce to the world
    LOG("%s", "MRM Sample App");

    // Now print out what we are doing
    LOG("Radio address: %s (USB)", radioAddr.c_str());
    LOG("%s", "Receiving scans directly from radio (no scan server)");

    // initialize the interface to the RCM
    check_or_exit(mrmIfInit(mrmIfUsb, radioAddr.c_str()) != OK,
            "Initialization failed");

    // TODO: (joshpfosi) Memory leak (see top of file)
    // make sure radio is in active mode
    check_or_exit(mrmSleepModeGet(&mode) != OK,
           "Time out waiting for sleep mode");

	// print sleep mode
    LOG("Radio sleep mode is %d", mode);
    if (mode != MRM_SLEEP_MODE_ACTIVE) {
        LOG("%s", "Changing sleep mode to Active");
        mrmSleepModeSet(MRM_SLEEP_MODE_ACTIVE);
    }

    // TODO: (joshpfosi) Memory leak (see top of file)
    // make sure radio is in MRM mode
    check_or_exit(mrmOpmodeGet(&mode) != OK,
            "Time out waiting for mode of operation");

	// print radio opmode
    LOG("Radio mode of operation is %d", mode);
    if (mode != MRM_OPMODE_MRM) {
        LOG("%s", "Changing radio mode to MRM");
        mrmOpmodeSet(MRM_OPMODE_MRM);
    }

    // TODO: (joshpfosi) Memory leak (see top of file)
    // retrieve config from MRM
    check_or_exit(mrmConfigGet(&config) != 0,
            "Time out waiting for config confirm");

    // modify config with user inputs
    config.scanStartPs          = userScanStart;
    config.scanEndPs            = userScanStop;
    config.baseIntegrationIndex = userBaseII;
    config.txGain               = userTxGain;
    config.codeChannel          = userCodeChannel;
    config.antennaMode          = userAntennaMode;
    config.scanResolutionBins   = userScanResolutionBins;

    // TODO: (joshpfosi) Memory leak (see top of file)
    // write updated config to radio
    check_or_exit(mrmConfigSet(&config) != 0,
	"Time out waiting for set config confirm");

    // print out configuration
    LOG("%s", "Configuration:");
    LOG("\tnodeId: %d", config.nodeId);
    LOG("\tscanStartPs: %d", config.scanStartPs);
    LOG("\tscanEndPs: %d", config.scanEndPs);
    LOG("\tscanResolutionBins: %d", config.scanResolutionBins);
    LOG("\tbaseIntegrationIndex: %d", config.baseIntegrationIndex);

    for (int i = 0 ; i < 4; i++) {
        LOG("\tsegment %d segmentNumSamples: %d", i, config.segmentNumSamples[i]);
        LOG("\tsegment %d segmentIntMult: %d", i, config.segmentIntMult[i]);
    }

    LOG("\tantennaMode: %d", config.antennaMode);
    LOG("\ttxGain: %d", config.txGain);
    LOG("\tcodeChannel: %d", config.codeChannel);

    // TODO: (joshpfosi) Memory leak (see top of file)
    // retrieve status info from MRM
    check_or_exit(mrmStatusInfoGet(&statusInfo) != 0,
            "Time out waiting for status info confirm");

    // print out status info
    LOG("%s", "Status Info:");
    LOG("\tPackage version: %s", statusInfo.packageVersionStr);
    LOG("\tApp version: %d.%d build %d", statusInfo.appVersionMajor,
            statusInfo.appVersionMinor, statusInfo.appVersionBuild);
    LOG("\tUWB Kernel version: %d.%d build %d",
            statusInfo.uwbKernelVersionMajor, statusInfo.uwbKernelVersionMinor,
            statusInfo.uwbKernelVersionBuild);
    LOG("\tFirmware version: %x/%x/%x ver %X", statusInfo.firmwareMonth,
            statusInfo.firmwareDay, statusInfo.firmwareYear,
            statusInfo.firmwareVersion);
    LOG("\tSerial number: %08X", statusInfo.serialNum);
    LOG("\tBoard revision: %c", statusInfo.boardRev);
    LOG("\tTemperature: %.2f degC", statusInfo.temperature / 4.0);
}

PulsOn::~PulsOn() {
    // NOTE: We do not raise exceptions in the destructor.
    // stop radio
    if (mrmControl(0, 0) != 0) {
        LOG("%s", "Time out waiting for control confirm");
    }

    mrmIfClose();
}

// Collects 1 radar pulse and populates `info`
//
// @raises CollectionError
void PulsOn::collect(pulsonInfo &info) {
    int timeoutMs = 200;

    LOG("\nScanning with scan count of %d and interval of %d (microseconds)",
            userScanCount, userScanInterval);

    // TODO: (joshpfosi) Memory leak (see top of file)
    check_or_exit(mrmControl(userScanCount, userScanInterval) != 0,
            "Time out waiting for control confirm");

    check_or_exit(mrmInfoGet(timeoutMs, &info) != 0, "mrmInfoGet ERR");
}

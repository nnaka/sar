#include "pulson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <time.h>

#include "debug.h"

using namespace std;

// NOTE: Most memory leaks marked below result from not calling
// `mrmSampleExit()` So `mrmIfClose()` is never called, and so the socket is
// forcibly released by the OS which can cause 'bind: Socket already in use'
// errors.

PulsOn::PulsOn(const string & radioAddr) :
	userBaseII(DEFAULT_BASEII),
    userScanStart(DEFAULT_SCAN_START),
    userScanStop(DEFAULT_SCAN_STOP),
    userScanCount(DEFAULT_SCAN_COUNT),
    userScanInterval(DEFAULT_SCAN_INTERVAL),
    userTxGain(DEFAULT_TX_GAIN)
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
	config.baseIntegrationIndex = userBaseII;
	config.scanStartPs          = userScanStart;
	config.scanEndPs            = userScanStop;
	config.txGain               = userTxGain;

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

// Collects 1 radar pulse
//
// @raises CollectionError
string PulsOn::collect() {
    int timeoutMs = 200;

    // raw and filtered scans and detection lists are sent in this struct
    mrmInfo info;

    LOG("\nScanning with scan count of %d and interval of %d (microseconds)",
            userScanCount, userScanInterval);

    // TODO: (joshpfosi) Memory leak (see top of file)
    check_or_exit(mrmControl(userScanCount, userScanInterval) != 0,
            "Time out waiting for control confirm");

    while (mrmInfoGet(timeoutMs, &info) == 0) {
        return processInfo(&info);
    }

    return "";
}

string PulsOn::processInfo(mrmInfo *info) {
    stringstream ss;

    switch (info->msg.scanInfo.msgType) {
        case MRM_DETECTION_LIST_INFO:
            // print number of detections and index
            // and magnitude of 1st detection
            LOG("DETECTION_LIST_INFO: msgId %d, numDetections %d, 1st detection"
                    "index: %d, 1st detection magnitude: %d",
                    info->msg.detectionList.msgId,
                    info->msg.detectionList.numDetections,
                    info->msg.detectionList.detections[0].index,
                    info->msg.detectionList.detections[0].magnitude);

            break;
		case MRM_FULL_SCAN_INFO:
            ss << "FULL_SCAN_INFO:"
               << " msgId " << info->msg.scanInfo.msgId << ","
               << " sourceId " << info->msg.scanInfo.sourceId << ","
               << " timestamp " << info->msg.scanInfo.timestamp << ","
               << " scanStartPs " << info->msg.scanInfo.scanStartPs << ","
               << " scanStopPs " << info->msg.scanInfo.scanStopPs << ","
               << " scanStepBins " << info->msg.scanInfo.scanStepBins << ","
               << " scanFiltering " << info->msg.scanInfo.scanFiltering << ","
               << " antennaId " << info->msg.scanInfo.antennaId << ","
               << " operationMode " << info->msg.scanInfo.operationMode << ","
               << " numSamplesTotal " << info->msg.scanInfo.numSamplesTotal << ","
               << " numMessagesTotal " << info->msg.scanInfo.numMessagesTotal;

            LOG("%s", ss.str().c_str());

			for (mrm_uint32_t i = 0;
                    i < info->msg.scanInfo.numSamplesTotal; ++i) {

                ss << ", " << info->scan[i];
            }

            free(info->scan);
            break;
        default:
            break;
    }

    return ss.str();
}

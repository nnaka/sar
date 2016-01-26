#include "pulson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using namespace std;

PulsOn::PulsOn(const string & radioAddr) :
    connected(false),
    userPrintInfo(true),
	userBaseII(DEFAULT_BASEII),
    userScanStart(DEFAULT_SCAN_START),
    userScanStop(DEFAULT_SCAN_STOP),
    userScanCount(DEFAULT_SCAN_COUNT),
    userScanInterval(DEFAULT_SCAN_INTERVAL),
    userTxGain(DEFAULT_TX_GAIN)
{
    int i, mode;

    // config and status strucs
    mrmMsg_GetStatusInfoConfirm statusInfo;

	// announce to the world
	printf("MRM Sample App\n\n");

    // Now print out what we are doing
    printf("Radio address: %s (USB)\n", radioAddr.c_str());
    printf("Receiving scans directly from radio (no scan server).\n");

    // initialize the interface to the RCM
    if (mrmIfInit(mrmIfUsb, radioAddr.c_str()) != OK) {
        printf("Initialization failed.\n");
        exit(0);
    }

    // make sure radio is in active mode
    if (mrmSleepModeGet(&mode) != OK) {
        printf("Time out waiting for sleep mode.\n");
        mrmSampleExit();
    }

	// print sleep mode
    printf("Radio sleep mode is %d.\n", mode);
    if (mode != MRM_SLEEP_MODE_ACTIVE) {
        printf("Changing sleep mode to Active.\n");
        mrmSleepModeSet(MRM_SLEEP_MODE_ACTIVE);
    }

    // make sure radio is in MRM mode
    if (mrmOpmodeGet(&mode) != OK) {
        printf("Time out waiting for mode of operation.\n");
        mrmSampleExit();
    }

	// print radio opmode
    printf("Radio mode of operation is %d.\n", mode);
    if (mode != MRM_OPMODE_MRM) {
        printf("Changing radio mode to MRM.\n");
        mrmOpmodeSet(MRM_OPMODE_MRM);
    }

    // retrieve config from MRM
    if (mrmConfigGet(&config) != 0) {
        printf("Time out waiting for config confirm.\n");
        mrmSampleExit();
    }

	// modify config with user inputs
	config.baseIntegrationIndex = userBaseII;
	config.scanStartPs = userScanStart;
	config.scanEndPs = userScanStop;
	config.txGain = userTxGain;

	// write updated config to radio
	if (mrmConfigSet(&config) != 0) {
		printf("Time out waiting for set config confirm.\n");
		mrmSampleExit();
	}

    // print out configuration
    printf("\nConfiguration:\n");
    printf("\tnodeId: %d\n", config.nodeId);
    printf("\tscanStartPs: %d\n", config.scanStartPs);
    printf("\tscanEndPs: %d\n", config.scanEndPs);
    printf("\tscanResolutionBins: %d\n", config.scanResolutionBins);
    printf("\tbaseIntegrationIndex: %d\n", config.baseIntegrationIndex);

    for (i = 0 ; i < 4; i++) {
        printf("\tsegment %d segmentNumSamples: %d\n", i, config.segmentNumSamples[i]);
        printf("\tsegment %d segmentIntMult: %d\n", i, config.segmentIntMult[i]);
    }

    printf("\tantennaMode: %d\n", config.antennaMode);
    printf("\ttxGain: %d\n", config.txGain);
    printf("\tcodeChannel: %d\n", config.codeChannel);

    // retrieve status info from MRM
    if (mrmStatusInfoGet(&statusInfo) != 0) {
        printf("Time out waiting for status info confirm.\n");
        mrmSampleExit();
    }

    // print out status info
    printf("\nStatus Info:\n");
    printf("\tPackage version: %s\n", statusInfo.packageVersionStr);
    printf("\tApp version: %d.%d build %d\n", statusInfo.appVersionMajor,
            statusInfo.appVersionMinor, statusInfo.appVersionBuild);
    printf("\tUWB Kernel version: %d.%d build %d\n", statusInfo.uwbKernelVersionMajor,
            statusInfo.uwbKernelVersionMinor, statusInfo.uwbKernelVersionBuild);
    printf("\tFirmware version: %x/%x/%x ver %X\n", statusInfo.firmwareMonth,
            statusInfo.firmwareDay, statusInfo.firmwareYear,
            statusInfo.firmwareVersion);
    printf("\tSerial number: %08X\n", statusInfo.serialNum);
    printf("\tBoard revision: %c\n", statusInfo.boardRev);
    printf("\tTemperature: %.2f degC\n\n", statusInfo.temperature/4.0);
}

PulsOn::~PulsOn() {
    // stop radio
    if (mrmControl(0, 0) != 0) {
        printf("Time out waiting for control confirm.\n");
        mrmSampleExit();
    }

    // perform cleanup
	printf("\n\nAll Done!\n");
    mrmSampleExit();
}

// Collects 1 GPS pulse
//
// @raises CollectionError
void PulsOn::collect() {
    // raw and filtered scans and detection lists are sent in this struct
    mrmInfo info;

    printf("\nScanning with scan count of %d and interval of %d (microseconds)\n", userScanCount, userScanInterval);
    int timeoutMs = 200;

    if (mrmControl(userScanCount, userScanInterval) != 0) {
        printf("Time out waiting for control confirm.\n");
        mrmSampleExit();
    }

    while (mrmInfoGet(timeoutMs, &info) == 0) {
        processInfo(&info, stdout, userPrintInfo);
    }
}

void PulsOn::mrmSampleExit(void) {
    if (connected) {
        mrmDisconnect();
    }

    mrmIfClose();
    exit(EXIT_FAILURE);
}

void PulsOn::processInfo(mrmInfo *info, FILE *fp, int printInfo) {
	unsigned int i;

    switch (info->msg.scanInfo.msgType) {
        case MRM_DETECTION_LIST_INFO:
            // print number of detections and index and magnitude of 1st detection
			if (printInfo)
                printf("DETECTION_LIST_INFO: msgId %d, numDetections %d, 1st detection index: %d, 1st detection magnitude: %d\n",
                        info->msg.detectionList.msgId,
                        info->msg.detectionList.numDetections,
                        info->msg.detectionList.detections[0].index,
                        info->msg.detectionList.detections[0].magnitude);

            break;

		case MRM_FULL_SCAN_INFO:
			if (printInfo)
				printf("FULL_SCAN_INFO: msgId %d, sourceId %d, timestamp %d, "
						"scanStartPs %d, scanStopPs %d, "
						"scanStepBins %d, scanFiltering %d, antennaId %d, "
						"operationMode %d, numSamplesTotal %d, numMessagesTotal %d\n",
						info->msg.scanInfo.msgId,
						info->msg.scanInfo.sourceId,
						info->msg.scanInfo.timestamp,
						info->msg.scanInfo.scanStartPs,
						info->msg.scanInfo.scanStopPs,
						info->msg.scanInfo.scanStepBins,
						info->msg.scanInfo.scanFiltering,
						info->msg.scanInfo.antennaId,
						info->msg.scanInfo.operationMode,
						info->msg.scanInfo.numSamplesTotal,
						info->msg.scanInfo.numMessagesTotal);

			// log scan message
			if (fp) {
				fprintf(fp, "%ld, MrmFullScanInfo, %d, %d, %d, %d, %d, %d, %d, %d, %u", (long)time(NULL), info->msg.scanInfo.msgId, info->msg.scanInfo.sourceId,
					info->msg.scanInfo.timestamp, info->msg.scanInfo.scanStartPs, info->msg.scanInfo.scanStopPs, info->msg.scanInfo.scanStepBins, 
						info->msg.scanInfo.scanFiltering, info->msg.scanInfo.antennaId, info->msg.scanInfo.numSamplesTotal);
				
				for (i = 0; i < info->msg.scanInfo.numSamplesTotal; i++)
					fprintf(fp, ", %d", info->scan[i]);
				fprintf(fp, "\n");
			}
            free(info->scan);
            break;

        default:
            break;
    }
}

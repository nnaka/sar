//_____________________________________________________________________________
//
// Copyright 2014 Time Domain Corporation
//
//
// mrmSampleApp.c
//
//   Sample code showing how to interface to P400 MRM module.
//
//   This code uses the functions in mrm.c to:
//      - connect to the radio via ethernet, usb, or serial
//      - get the MRM configuration from the radio and print it
//      - get the status/info from the radio and print it
//      - start MRM and receive scans from it according to user parameters
//
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// #includes 
//_____________________________________________________________________________

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mrmIf.h"
#include "mrm.h"


//_____________________________________________________________________________
//
// #defines 
//_____________________________________________________________________________

#define DEFAULT_RADIO_IP			"192.168.1.100"
#define DEFAULT_BASEII				12
#define DEFAULT_SCAN_START			10000
#define DEFAULT_SCAN_STOP			39297
#define DEFAULT_SCAN_COUNT			5
#define DEFAULT_SCAN_INTERVAL		125000
#define DEFAULT_TX_GAIN				63


//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// static data
//_____________________________________________________________________________

static int haveServiceIp, connected;

//_____________________________________________________________________________
//
// local functions
//_____________________________________________________________________________

//_____________________________________________________________________________
//
// processInfo - process info messages from MRM
//_____________________________________________________________________________

static void processInfo(mrmInfo *info, FILE *fp, int printInfo)
{
	unsigned int i;

    switch (info->msg.scanInfo.msgType)
    {
        case MRM_DETECTION_LIST_INFO:
            // print number of detections and index and magnitude of 1st detection
			if (printInfo)
                printf("DETECTION_LIST_INFO: msgId %d, numDetections %d, 1st detection index: %d, 1st detection magnitude: %d\n",
                        info->msg.detectionList.msgId,
                        info->msg.detectionList.numDetections,
                        info->msg.detectionList.detections[0].index,
                        info->msg.detectionList.detections[0].magnitude);

			// log detection list message
            if (haveServiceIp && fp)
            {
    			fprintf(fp, "%ld, MrmDetectionListInfo, %d, %d", (long)time(NULL), info->msg.detectionList.msgId, info->msg.detectionList.numDetections);
				for (i = 0; i < info->msg.detectionList.numDetections; i++)
					fprintf(fp, ", %d, %d", info->msg.detectionList.detections[i].index, info->msg.detectionList.detections[i].magnitude);
				fprintf(fp, "\n");
            }
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

//_____________________________________________________________________________
//
// usage - print command line options
//_____________________________________________________________________________

static void usage(void)
{
	printf("  USAGE: mrmsampleApp [OPTIONS] -u <USB Com Port> | -e <Ethernet IP Address> | -o <Serial COM Port>\n");
	printf("OPTIONS:    -m <MRM Service IP address> (Default = No Server, connect directly to radio)\n");
	printf("            -b <Base Integration Index> (Default = %d)\n", DEFAULT_BASEII);
	printf("            -s <Scan Start (picoseconds)> (Default = %d)\n", DEFAULT_SCAN_START);
	printf("            -p <Scan Stop (picoseconds)> (Default = %d)\n", DEFAULT_SCAN_STOP);
	printf("            -c <Scan Counts> (Default = %d)\n", DEFAULT_SCAN_COUNT);
	printf("            -i <Scan Interval (microseconds)> (Default = %d)\n", DEFAULT_SCAN_INTERVAL);
	printf("            -g <TX Gain> (Default = %d)\n", DEFAULT_TX_GAIN);
	printf("            -x Print Info Messages <1 = On/0 = Off> (Default = On(1))\n");
	printf("            -l <Log File Prefix> (Default = No Log File)\n");
    exit(0);
}

//_____________________________________________________________________________
//
// mrmSampleExit - close interfaces and exit sample app
//_____________________________________________________________________________

static void mrmSampleExit(void)
{
    if (connected)
        mrmDisconnect();
    mrmIfClose();
    exit(0);
}


//_____________________________________________________________________________
//
// main - sample app entry point
//_____________________________________________________________________________

int main(int argc, char *argv[])
{
	FILE *fp = NULL;
    char *radioAddr = DEFAULT_RADIO_IP, *serviceIp;
    int haveRadioAddr = 0;
    int i, timeoutMs, mode;
	int userBaseII, userScanStart, userScanStop, userScanCount, userScanInterval, userTxGain, userPrintInfo, userLogging;
	char userLogPrefix[256];
	char logFileName[256];
	mrmIfType mrmIf;

    // config and status strucs
    mrmConfiguration config;
    mrmMsg_GetStatusInfoConfirm statusInfo;

    // filter settings
    mrmFilterConfig filterConfig;

    // raw and filtered scans and detection lists are sent in this struct
    mrmInfo info;

	// announce to the world
	printf("MRM Sample App\n\n");

	// Set Defaults
	userBaseII = DEFAULT_BASEII;
	userScanStart = DEFAULT_SCAN_START;
	userScanStop = DEFAULT_SCAN_STOP;
	userScanCount = DEFAULT_SCAN_COUNT;
	userScanInterval = DEFAULT_SCAN_INTERVAL;
	userTxGain = DEFAULT_TX_GAIN;
	userPrintInfo = 1;
	userLogging = 0;

	//
	// NOTE
	//   Need to add code to validate command line arguments
	//

	// process command line arguments
	if (argc < 3) {
		usage();
		exit(0);
	}

    for (i = 1; i < argc; )
    {
        // see if arg starts with a hyphen (meaning it's an option)
        if (*argv[i] == '-')
        {
            switch (argv[i++][1])
            {
				// USB
				case 'u':
			        mrmIf = mrmIfUsb;
					haveRadioAddr = 1;
					radioAddr = argv[i++];
					break;

				// Ethernet
				case 'e':
			        mrmIf = mrmIfIp;
					haveRadioAddr = 1;
					radioAddr = argv[i++];
					break;

				// Serial COM
				case 'o':
			        mrmIf = mrmIfSerial;
					haveRadioAddr = 1;
					radioAddr = argv[i++];
					break;

				// Base Integration Index
				case 'b':
					userBaseII = atoi(argv[i++]);
					break;

				// Scan Start
				case 's':
					userScanStart = atoi(argv[i++]);
					break;
					
				// Scan Stop
				case 'p':
					userScanStop = atoi(argv[i++]);
					break;

				// Scan Count
				case 'c':
					userScanCount = atoi(argv[i++]);
					break;

				// Scan Interval
				case 'i':
					userScanInterval = atoi(argv[i++]);
					break;

				// TX Gain
				case 'g':
					userTxGain = atoi(argv[i++]);
					break;

                // Service IP address
                case 'm':
                    haveServiceIp = 1;
                    serviceIp = argv[i++];
                    break;

				// Print Info Messages
				case 'x':
					userPrintInfo = atoi(argv[i++]);
					break;
					
				// Log file
				case 'l':
					strcpy(userLogPrefix, argv[i++]);
					userLogging = 1;
					break;

                default:
                    printf("Unknown option letter.\n\n");
                    usage();
                    break;
            }
        }
    }

    if (!haveRadioAddr)
        usage();

    // Now print out what we are doing
    printf("Radio address: %s (%s)\n", radioAddr, mrmIf == mrmIfUsb ? "USB" :
            (mrmIf == mrmIfIp ? "Ethernet" : "Serial"));
    if (haveServiceIp)
        printf("Using MRM service at IP address: %s\n", serviceIp);
    else
        printf("Receiving scans directly from radio (no scan server).\n");

    // If using service, connect to it
    if (haveServiceIp)
    {
        mrm_uint32_t status;

        if (mrmIfInit(mrmIfIp, serviceIp) != OK)
        {
            printf("Failed to connect to service (bad IP address?).\n");
            exit(0);
        }
        // connect to radio
        if ((mrmConnect(mrmIf, radioAddr, &status) != OK) ||
                (status != MRM_SERVER_CONNECTIONSTATUSCODE_CONNECTED))
        {
            printf("Unable to connect to radio through service.\n");
            mrmSampleExit();
        }
        connected = 1;
    } else
    {
        // initialize the interface to the RCM
        if (mrmIfInit(mrmIf, radioAddr) != OK)
        {
            printf("Initialization failed.\n");
            exit(0);
        }
    }

    // make sure radio is in active mode
    if (mrmSleepModeGet(&mode) != OK)
    {
        printf("Time out waiting for sleep mode.\n");
        mrmSampleExit();
    }

	// print sleep mode
    printf("Radio sleep mode is %d.\n", mode);
    if (mode != MRM_SLEEP_MODE_ACTIVE)
    {
        printf("Changing sleep mode to Active.\n");
        mrmSleepModeSet(MRM_SLEEP_MODE_ACTIVE);
    }

    // make sure radio is in MRM mode
    if (mrmOpmodeGet(&mode) != OK)
    {
        printf("Time out waiting for mode of operation.\n");
        mrmSampleExit();
    }

	// print radio opmode
    printf("Radio mode of operation is %d.\n", mode);
    if (mode != MRM_OPMODE_MRM)
    {
        printf("Changing radio mode to MRM.\n");
        mrmOpmodeSet(MRM_OPMODE_MRM);
    }

    // retrieve config from MRM
    if (mrmConfigGet(&config) != 0)
    {
        printf("Time out waiting for config confirm.\n");
        mrmSampleExit();
    }

	// modify config with user inputs
	config.baseIntegrationIndex = userBaseII;
	config.scanStartPs = userScanStart;
	config.scanEndPs = userScanStop;
	config.txGain = userTxGain;

	// write updated config to radio
	if (mrmConfigSet(&config) != 0)
	{
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
    for (i = 0 ; i < 4; i++)
    {
        printf("\tsegment %d segmentNumSamples: %d\n", i, config.segmentNumSamples[i]);
        printf("\tsegment %d segmentIntMult: %d\n", i, config.segmentIntMult[i]);
    }
    printf("\tantennaMode: %d\n", config.antennaMode);
    printf("\ttxGain: %d\n", config.txGain);
    printf("\tcodeChannel: %d\n", config.codeChannel);

    if (haveServiceIp)
    {
        // now get the filter config from MRM
        if (mrmFilterConfigGet(&filterConfig) != 0)
        {
            printf("Time out waiting for filter config confirm.\n");
            mrmSampleExit();
        }

        // print out filter configuration
        printf("\nFilter Configuration:\n");
        printf("\tfilterFlags: %d\n", filterConfig.filterFlags);
        printf("\tmotionFilterIndex: %d\n", filterConfig.motionFilterIndex);
        printf("\tdetectionListThresholdMult: %d\n", filterConfig.detectionListThresholdMult);
    }

    // retrieve status info from MRM
    if (mrmStatusInfoGet(&statusInfo) != 0)
    {
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

	// start logging
	if (userLogging)
	{
		sprintf(logFileName, "%sMRM_Scans.csv", userLogPrefix);
		printf("\nLogging to file %s\n\n", logFileName);
		fp = fopen(logFileName, "w");
		if (fp)
		{
			// Log Configuration
			fprintf(fp, "Timestamp, Config, NodeId, ScanStartPs, ScanStopPs, ScanResolutionBins, BaseIntegrationIndex, Segment1NumSamples, Segment2NumSamples, Segment3NumSamples, Segment4NumSamples, Segment1AdditionalIntegration, Segment2AdditionalIntegration, Segment3AdditionalIntegration, Segment4AdditionalIntegration, AntennaMode, TransmitGain, CodeChannel\n");
			fprintf(fp, "%ld, Config, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
					(long)time(NULL), config.nodeId, config.scanStartPs, config.scanEndPs, config.scanResolutionBins, config.baseIntegrationIndex,
					config.segmentNumSamples[0], config.segmentNumSamples[1], config.segmentNumSamples[2], config.segmentNumSamples[3],
					config.segmentIntMult[0], config.segmentIntMult[1], config.segmentIntMult[2], config.segmentIntMult[3], 
					config.antennaMode, config.txGain, config.codeChannel);

			// Log Control
			fprintf(fp, "Timestamp, Control, ScanCount, IntervalTimeMicroseconds\n");
			fprintf(fp, "%ld, Control, %d, %d\n", (long)time(NULL), userScanCount, userScanInterval);

			// Write Full Scan Header
			fprintf(fp, "Timestamp, MrmFullScanInfo, MessageId, SourceId, Timestamp, ScanStartPs, ScanStopPs, ScanStepBins, Filtering, AntennaId, NumSamplesTotal, ScanData\n");

            // If using service, print Detection List header
            if (haveServiceIp)
    			fprintf(fp, "Timestamp, MrmDetectionListInfo, MessageId, NumDetections, DetectionData\n");
		}
		else
		{
			printf("Error opening %s\n", logFileName);
			mrmSampleExit();
		}
	}

    //
    // Ready to start collecting scans...
	//

    // Start scanning on the MRM
    printf("\nScanning with scan count of %d and interval of %d (microseconds)\n", userScanCount, userScanInterval);
    timeoutMs = 200;
    if (mrmControl(userScanCount, userScanInterval) != 0)
    {
        printf("Time out waiting for control confirm.\n");
        mrmSampleExit();
    }
    while (mrmInfoGet(timeoutMs, &info) == 0)
        processInfo(&info, fp, userPrintInfo);

    // stop radio
    if (mrmControl(0, 0) != 0)
    {
        printf("Time out waiting for control confirm.\n");
        mrmSampleExit();
    }

    // perform cleanup
	printf("\n\nAll Done!\n");
	if (fp)
		fclose(fp);
    mrmSampleExit();
    return 0;
}


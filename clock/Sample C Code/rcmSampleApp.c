//_____________________________________________________________________________
//
// Copyright 2011-2 Time Domain Corporation
//
//
// rcmSampleApp.c
//
//   Sample code showing how to interface to P400 RCM module.
//
//   This code uses the functions in rcm.c to:
//      - make sure the RCM is awake and in the correct mode
//      - get the configuration from the RCM and print it
//      - get the status/info from the RCM and print it
//      - range to another RCM node
//      - broadcast that range in a data packet
//
// This sample can communicate with the RCM over Ethernet, the 3.3V serial port,
// or the USB interface (which acts like a serial port).
//
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// #includes 
//_____________________________________________________________________________

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rcmIf.h"
#include "rcm.h"


//_____________________________________________________________________________
//
// #defines 
//_____________________________________________________________________________

#define DEFAULT_DEST_NODE_ID    101


//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// static data
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// local function prototypes
//_____________________________________________________________________________

void usage(void)
{
    printf("usage: rcmSampleApp -i <IP address> | -s <COM port> | -u <USB COM port>\n");
    printf("\nTo connect to radio at IP address 192.168.1.100 via Ethernet:\n");
    printf("\trcmSampleApp -i 192.168.1.100\n");
    printf("\nTo connect to radio's serial port using USB-to-TTL serial converter at COM3:\n");
    printf("\trcmSampleApp -s COM3\n");
    printf("\nTo connect to radio's USB port at COM10:\n");
    printf("\trcmSampleApp -u COM10\n");
    exit(0);
}


//_____________________________________________________________________________
//
// main - sample app entry point
//_____________________________________________________________________________

int main(int argc, char *argv[])
{
    char str[100];
    int destNodeId=DEFAULT_DEST_NODE_ID;
    int status;
    rcmIfType   rcmIf;
    rcmConfiguration config;
    rcmMsg_GetStatusInfoConfirm statusInfo;
    rcmMsg_RangeInfo rangeInfo;
    rcmMsg_DataInfo dataInfo;
    rcmMsg_ScanInfo scanInfo;
    rcmMsg_FullScanInfo fullScanInfo;

	printf("RCM Sample App\n\n");

    // check command line arguments
    if (argc != 3)
        usage();

    if (!strcmp(argv[1], "-i"))
        rcmIf = rcmIfIp;
    else if (!strcmp(argv[1], "-s"))
        rcmIf = rcmIfSerial;
    else if (!strcmp(argv[1], "-u"))
        rcmIf = rcmIfUsb;
    else
        usage();

    // initialize the interface to the RCM
    if (rcmIfInit(rcmIf, argv[2]) != OK)
    {
        printf("Initialization failed.\n");
        exit(0);
    }

    // Make sure RCM is awake
    if (rcmSleepModeSet(RCM_SLEEP_MODE_ACTIVE) != 0)
    {
        printf("Time out waiting for sleep mode set.\n");
        exit(0);
    }

    // Make sure opmode is RCM
    if (rcmOpModeSet(RCM_OPMODE_RCM) != 0)
    {
        printf("Time out waiting for opmode set.\n");
        exit(0);
    }

    // execute Built-In Test - verify that radio is healthy
    if (rcmBit(&status) != 0)
    {
        printf("Time out waiting for BIT.\n");
        exit(0);
    }

    if (status != OK)
    {
        printf("Built-in test failed - status %d.\n", status);
        exit(0);
    }
    else
    {
        printf("Radio passes built-in test.\n\n");
    }

    // retrieve config from RCM
    if (rcmConfigGet(&config) != 0)
    {
        printf("Time out waiting for config confirm.\n");
        exit(0);
    }

    // print out configuration
    printf("Configuration:\n");
    printf("\tnodeId: %d\n", config.nodeId);
    printf("\tintegrationIndex: %d\n", config.integrationIndex);
    printf("\tantennaMode: %d\n", config.antennaMode);
    printf("\tcodeChannel: %d\n", config.codeChannel);
    printf("\telectricalDelayPsA: %d\n", config.electricalDelayPsA);
    printf("\telectricalDelayPsB: %d\n", config.electricalDelayPsB);
    printf("\tflags: 0x%X\n", config.flags);
    printf("\ttxGain: %d\n", config.txGain);

    // retrieve status/info from RCM
    if (rcmStatusInfoGet(&statusInfo) != 0)
    {
        printf("Time out waiting for status info confirm.\n");
        exit(0);
    }

    // print out status/info
    printf("\nStatus/Info:\n");
    printf("\tRCM version: %d.%d build %d\n", statusInfo.appVersionMajor,
            statusInfo.appVersionMinor, statusInfo.appVersionBuild);
    printf("\tUWB Kernel version: %d.%d build %d\n", statusInfo.uwbKernelVersionMajor,
            statusInfo.uwbKernelVersionMinor, statusInfo.uwbKernelVersionBuild);
    printf("\tFirmware version: %x/%x/%x ver %X\n", statusInfo.firmwareMonth,
            statusInfo.firmwareDay, statusInfo.firmwareYear,
            statusInfo.firmwareVersion);
    printf("\tSerial number: %08X\n", statusInfo.serialNum);
    printf("\tBoard revision: %c\n", statusInfo.boardRev);
    printf("\tTemperature: %.2f degC\n\n", statusInfo.temperature/4.0);

    // enter loop ranging to a node and broadcasting the resulting range
    while (1)
    {
        // get ranging target node ID from user
        printf("Enter node ID of radio to range to or q to quit [%d]: ", destNodeId);
        fgets(str, sizeof(str), stdin);
        if (*str == 'q')
            break;
        if (strlen(str) > 1)
            destNodeId = atoi(str);

        // Determine range to a radio. May also get data and scan packets.
        if (rcmRangeTo(destNodeId, RCM_ANTENNAMODE_TXA_RXA, 0, NULL,
                &rangeInfo, &dataInfo, &scanInfo, &fullScanInfo) == 0)
        {
            // we always get a range info packet
            printf("RANGE_INFO: responder %d, msg ID %u, range status %d, "
                    "stopwatch %d ms, channelRiseTime %d, vPeak %d, measurement type %d\n",
                    rangeInfo.responderId, rangeInfo.msgId, rangeInfo.rangeStatus,
                    rangeInfo.stopwatchTime, rangeInfo.channelRiseTime, rangeInfo.vPeak,
                    rangeInfo.rangeMeasurementType);

            // The RANGE_INFO can provide multiple types of ranges
            if (rangeInfo.rangeMeasurementType & RCM_RANGE_TYPE_PRECISION)
            {
                printf("Precision range: %d mm, error estimate %d mm\n",
                        rangeInfo.precisionRangeMm, rangeInfo.precisionRangeErrEst);
            }

            if (rangeInfo.rangeMeasurementType & RCM_RANGE_TYPE_COARSE)
            {
                printf("Coarse range: %d mm, error estimate %d mm\n",
                        rangeInfo.coarseRangeMm, rangeInfo.coarseRangeErrEst);
            }

            if (rangeInfo.rangeMeasurementType & RCM_RANGE_TYPE_FILTERED)
            {
                printf("Filtered range: %d mm, error estimate %d mm\n",
                        rangeInfo.filteredRangeMm, rangeInfo.filteredRangeErrEst);
                printf("Filtered velocity: %d mm/s, error estimate %d mm/s\n",
                        rangeInfo.filteredRangeVel, rangeInfo.filteredRangeVelErrEst);
            }


            // only get a data info packet if the responder sent data
            // dataSize will be non-zero if we there is data
            if (dataInfo.dataSize)
                printf("DATA_INFO from node %d: msg ID %u, channelRiseTime %d, vPeak %d, %d bytes\ndata: %*s\n",
                        dataInfo.sourceId, dataInfo.msgId, dataInfo.channelRiseTime, dataInfo.vPeak,
                        dataInfo.dataSize, dataInfo.dataSize, dataInfo.data);

            // only get a scan info packet if the SEND_SCAN bit is set in the config
            // numSamples will be non-zero if there is scan data
            // we don't do anything with the scan data itself here
            if (scanInfo.numSamples)
                printf("SCAN_INFO from node %d: msg ID %u, %d samples, channelRiseTime %d, vPeak %d\n",
                        scanInfo.sourceId, scanInfo.msgId, scanInfo.numSamples,
                        scanInfo.channelRiseTime, scanInfo.vPeak);

            // only get a full scan info packet if the FULL_SCAN bit is set in the config
            // numSamplesInMessage will be non-zero if there is scan data
            // we don't do anything with the scan data itself here
            if (fullScanInfo.numSamplesInMessage)
                printf("FULL_SCAN_INFO from node %d: msg ID %u, %d samples, channelRiseTime %d, vPeak %d\n",
                        fullScanInfo.sourceId, fullScanInfo.msgId, fullScanInfo.numSamplesInMessage,
                        fullScanInfo.channelRiseTime, fullScanInfo.vPeak);

            // a rangeStatus of 0 means the range was calculated successfully
            if (rangeInfo.rangeStatus == 0)
            {
                // now broadcast the range in a data packet
                sprintf(str, "The range from %d to %d is %d mm.",
                        config.nodeId, destNodeId,
                        rangeInfo.precisionRangeMm);
                rcmDataSend(RCM_ANTENNAMODE_TXA_RXA, strlen(str), str);
            }
        }
    }

    // perform cleanup
    rcmIfClose();
    return 0;
}


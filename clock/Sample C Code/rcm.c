//_____________________________________________________________________________
//
// Copyright 2011-2 Time Domain Corporation
//
//
// rcm.c
//
//   A collection of functions to communicate with an RCM.
//
//
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// #includes 
//_____________________________________________________________________________

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
#include "winsock2.h"
#else // linux
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "rcmIf.h"
#include "rcm.h"


//_____________________________________________________________________________
//
// #defines 
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________

typedef union
{
    rcmMsg_RangeInfo rangeInfo;
    rcmMsg_DataInfo dataInfo;
    rcmMsg_ScanInfo scanInfo;
    rcmMsg_FullScanInfo fullScanInfo;
} infoMsgs_t;


//_____________________________________________________________________________
//
// static data
//_____________________________________________________________________________

static int msgIdCount;


//_____________________________________________________________________________
//
// Private function prototypes 
//_____________________________________________________________________________



//_____________________________________________________________________________
//
// rcmBit - execute Built-In Test
//_____________________________________________________________________________

int rcmBit(int *status)
{
    rcmMsg_BitRequest request;
    rcmMsg_BitConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_BIT_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if (confirm.msgType == RCM_BIT_CONFIRM)
        {
            *status = confirm.status;
            retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmConfigGet - get rcm configuration from radio
//_____________________________________________________________________________

int rcmConfigGet(rcmConfiguration *config)
{
    rcmMsg_GetConfigRequest request;
    rcmMsg_GetConfigConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_GET_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(rcmMsg_GetConfigConfirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(rcmMsg_GetConfigConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == RCM_GET_CONFIG_CONFIRM)
        {
            // copy config from message to caller's structure
            memcpy(config, &confirm.config, sizeof(*config));
            // Handle byte ordering
            config->nodeId = ntohl(config->nodeId);
            config->integrationIndex = ntohs(config->integrationIndex);
            config->electricalDelayPsA = ntohl(config->electricalDelayPsA);
            config->electricalDelayPsB = ntohl(config->electricalDelayPsB);
            config->flags = ntohs(config->flags);

            // milliseconds since radio boot
            confirm.timestamp = ntohl(confirm.timestamp);

            // status code
            confirm.status = ntohl(confirm.status);
            // only return OK if status is OK
            if (confirm.status == OK)
                retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmConfigSet - set RCM configuration in radio
//_____________________________________________________________________________

int rcmConfigSet(rcmConfiguration *config)
{
    rcmMsg_SetConfigRequest request;
    rcmMsg_SetConfigConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_SET_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);
    memcpy(&request.config, config, sizeof(*config));

    // Handle byte ordering in config struct
    request.config.nodeId = htonl(config->nodeId);
    request.config.integrationIndex = htons(config->integrationIndex);
    request.config.electricalDelayPsA = htonl(config->electricalDelayPsA);
    request.config.electricalDelayPsB = htonl(config->electricalDelayPsB);
    request.config.flags = htons(config->flags);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if (confirm.msgType == RCM_SET_CONFIG_CONFIRM &&
                confirm.status == OK)
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmOpModeSet - set RCM operational mode
//_____________________________________________________________________________

int rcmOpModeSet(int opMode)
{
    rcmMsg_SetOpmodeRequest request;
    rcmMsg_SetOpmodeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_SET_OPMODE_REQUEST);
	request.msgId = htons(msgIdCount++);
    request.opMode = htonl(opMode);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if (confirm.msgType == RCM_SET_OPMODE_CONFIRM &&
                confirm.status == OK)
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmSleepModeSet - set RCM sleep mode
//_____________________________________________________________________________

int rcmSleepModeSet(int sleepMode)
{
    rcmMsg_SetSleepModeRequest request;
    rcmMsg_SetSleepModeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_SET_SLEEP_MODE_REQUEST);
	request.msgId = htons(msgIdCount++);
    request.sleepMode = htonl(sleepMode);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if (confirm.msgType == RCM_SET_SLEEP_MODE_CONFIRM &&
                confirm.status == OK)
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmStatusInfoGet - retrieve RCM status from radio
//_____________________________________________________________________________

int rcmStatusInfoGet(rcmMsg_GetStatusInfoConfirm *confirm)
{
    rcmMsg_GetStatusInfoRequest request;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_GET_STATUS_INFO_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(confirm, sizeof(rcmMsg_GetStatusInfoConfirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(rcmMsg_GetStatusInfoConfirm))
    {
        // Handle byte ordering
        confirm->msgType = ntohs(confirm->msgType);
        confirm->msgId = ntohs(confirm->msgId);

        // is this the correct message type?
        if (confirm->msgType == RCM_GET_STATUS_INFO_CONFIRM)
        {
            // Handle byte ordering
            confirm->appVersionBuild = ntohs(confirm->appVersionBuild);
            confirm->uwbKernelVersionBuild = ntohs(confirm->uwbKernelVersionBuild);
            confirm->serialNum = ntohl(confirm->serialNum);
            confirm->temperature = ntohl(confirm->temperature);

            // status code
            confirm->status = ntohl(confirm->status);
            // only return OK if status is OK
            if (confirm->status == OK)
                retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmRangeTo - range to another RCM module
//_____________________________________________________________________________

int rcmRangeTo(int destNodeId, int antennaMode, int dataSize, char *data,
        rcmMsg_RangeInfo *rangeInfo, rcmMsg_DataInfo *dataInfo, rcmMsg_ScanInfo *scanInfo,
        rcmMsg_FullScanInfo *fullScanInfo)
{
    infoMsgs_t infoMsgs;
    rcmMsg_SendRangeRequest request;
    rcmMsg_SendRangeRequestConfirm confirm;
    int retVal = ERR, numBytes;
    unsigned i;

    // create request message
	request.msgType = htons(RCM_SEND_RANGE_REQUEST);
	request.msgId = htons(msgIdCount++);
    request.responderId = htonl(destNodeId);
    request.antennaMode = antennaMode;
    request.dataSize = htons(dataSize);
    // make sure there isn't too much data
    if (dataSize > RCM_USER_DATA_LENGTH)
        dataSize = RCM_USER_DATA_LENGTH;
    // copy data into message
    memcpy(request.data, data, dataSize);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
    numBytes = sizeof(request) - RCM_USER_DATA_LENGTH + dataSize;
	rcmIfSendPacket(&request, numBytes);

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == RCM_SEND_RANGE_REQUEST_CONFIRM)
        {
            // check status code
            confirm.status = ntohl(confirm.status);
            if (confirm.status == OK)
            {
                retVal = OK;

                // clear out caller's info structs
                memset(rangeInfo, 0, sizeof(*rangeInfo));
                memset(dataInfo, 0, sizeof(*dataInfo));
                memset(scanInfo, 0, sizeof(*scanInfo));
                memset(fullScanInfo, 0, sizeof(*fullScanInfo));
                rangeInfo->rangeStatus = RCM_RANGE_STATUS_TIMEOUT;

                // Collect any info messages
                // We will always get a rangeInfo, maybe a dataInfo and/or scanInfo also
                while ((numBytes = rcmIfGetPacket(&infoMsgs, sizeof(infoMsgs))) > 0)
                {
                    // make sure this info message has the same msgId as the request
                    // the msgId is in the same place in all structs
                    if (ntohs(infoMsgs.rangeInfo.msgId) == msgIdCount - 1)
                    {
                        switch(ntohs(infoMsgs.rangeInfo.msgType))
                        {
                            case RCM_RANGE_INFO:
                                // copy message to caller's struct
                                memcpy(rangeInfo, &infoMsgs.rangeInfo, sizeof(*rangeInfo));
                                // handle byte ordering
                                rangeInfo->msgType = ntohs(rangeInfo->msgType);
                                rangeInfo->msgId = ntohs(rangeInfo->msgId);
                                rangeInfo->responderId = ntohl(rangeInfo->responderId);
                                rangeInfo->stopwatchTime = ntohs(rangeInfo->stopwatchTime);
                                rangeInfo->precisionRangeMm = ntohl(rangeInfo->precisionRangeMm);
                                rangeInfo->coarseRangeMm = ntohl(rangeInfo->coarseRangeMm);
                                rangeInfo->filteredRangeMm = ntohl(rangeInfo->filteredRangeMm);
                                rangeInfo->precisionRangeErrEst = ntohs(rangeInfo->precisionRangeErrEst);
                                rangeInfo->coarseRangeErrEst = ntohs(rangeInfo->coarseRangeErrEst);
                                rangeInfo->filteredRangeErrEst = ntohs(rangeInfo->filteredRangeErrEst);
                                rangeInfo->filteredRangeVel = ntohs(rangeInfo->filteredRangeVel);
                                rangeInfo->filteredRangeVelErrEst = ntohs(rangeInfo->filteredRangeVelErrEst);
                                rangeInfo->reqLEDFlags = ntohs(rangeInfo->reqLEDFlags);
                                rangeInfo->respLEDFlags = ntohs(rangeInfo->respLEDFlags);
                                rangeInfo->channelRiseTime = ntohs(rangeInfo->channelRiseTime);
                                rangeInfo->vPeak = ntohs(rangeInfo->vPeak);
                                rangeInfo->coarseTOFInBins = ntohl(rangeInfo->coarseTOFInBins);
                                rangeInfo->timestamp = ntohl(rangeInfo->timestamp);
                                break;
                            case RCM_DATA_INFO:
                                // copy message to caller's struct
                                memcpy(dataInfo, &infoMsgs.dataInfo, sizeof(*dataInfo));
                                // handle byte ordering
                                dataInfo->msgType = ntohs(dataInfo->msgType);
                                dataInfo->msgId = ntohs(dataInfo->msgId);
                                dataInfo->sourceId = ntohl(dataInfo->sourceId);
                                dataInfo->channelRiseTime = ntohs(dataInfo->channelRiseTime);
                                dataInfo->vPeak = ntohs(dataInfo->vPeak);
                                dataInfo->timestamp = ntohl(dataInfo->timestamp);
                                dataInfo->dataSize = ntohs(dataInfo->dataSize);
                                break;
                            case RCM_SCAN_INFO:
                                // copy message to caller's struct
                                memcpy(scanInfo, &infoMsgs.scanInfo, sizeof(*scanInfo));
                                // handle byte ordering
                                scanInfo->msgType = ntohs(scanInfo->msgType);
                                scanInfo->msgId = ntohs(scanInfo->msgId);
                                scanInfo->sourceId = ntohl(scanInfo->sourceId);
                                scanInfo->LEDflags = ntohs(scanInfo->LEDflags);
                                scanInfo->channelRiseTime = ntohs(scanInfo->channelRiseTime);
                                scanInfo->vPeak = ntohs(scanInfo->vPeak);
                                scanInfo->timestamp = ntohl(scanInfo->timestamp);
                                scanInfo->ledIndex = ntohl(scanInfo->ledIndex);
                                scanInfo->lockspotOffset = ntohl(scanInfo->lockspotOffset);
                                scanInfo->numSamples = ntohl(scanInfo->numSamples);
                                for (i = 0; i < scanInfo->numSamples; i++)
                                    scanInfo->samples[i] = ntohl(scanInfo->samples[i]);
                                break;
                            case RCM_FULL_SCAN_INFO:
                                //
                                // NOTE: this only returns the last RCM_FULL_SCAN_INFO packet
                                //
                                // copy message to caller's struct
                                memcpy(fullScanInfo, &infoMsgs.fullScanInfo, sizeof(*fullScanInfo));
                                // handle byte ordering
                                fullScanInfo->msgType = ntohs(fullScanInfo->msgType);
                                fullScanInfo->msgId = ntohs(fullScanInfo->msgId);
                                fullScanInfo->sourceId = ntohl(fullScanInfo->sourceId);
                                fullScanInfo->timestamp = ntohl(fullScanInfo->timestamp);
                                fullScanInfo->channelRiseTime = ntohs(fullScanInfo->channelRiseTime);
                                fullScanInfo->vPeak = ntohs(fullScanInfo->vPeak);
                                fullScanInfo->ledIndex = ntohl(fullScanInfo->ledIndex);
                                fullScanInfo->lockspotOffset = ntohl(fullScanInfo->lockspotOffset);
                                fullScanInfo->scanStartPs = ntohl(fullScanInfo->scanStartPs);
                                fullScanInfo->scanStopPs = ntohl(fullScanInfo->scanStopPs);
                                fullScanInfo->scanStepBins = ntohs(fullScanInfo->scanStepBins);
                                fullScanInfo->numSamplesInMessage = ntohs(fullScanInfo->numSamplesInMessage);
                                fullScanInfo->numSamplesTotal = ntohl(fullScanInfo->numSamplesTotal);
                                fullScanInfo->messageIndex = ntohs(fullScanInfo->messageIndex);
                                fullScanInfo->numMessagesTotal = ntohs(fullScanInfo->numMessagesTotal);
                                for (i = 0; i < fullScanInfo->numSamplesInMessage; i++)
                                    fullScanInfo->scan[i] = ntohl(fullScanInfo->scan[i]);
                                break;
                        }
						
						// We get RANGE_INFO last.
						if(ntohs(infoMsgs.rangeInfo.msgType) == RCM_RANGE_INFO)
							break;
                    }
                }
            }
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// rcmDataSend - broadcast a data-only packet
//_____________________________________________________________________________

int rcmDataSend(int antennaMode, int dataSize, char *data)
{
    rcmMsg_SendDataRequest request;
    rcmMsg_SendDataConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(RCM_SEND_DATA_REQUEST);
	request.msgId = htons(msgIdCount++);
    request.antennaMode = antennaMode;
    request.dataSize = htons(dataSize);
    // make sure there isn't too much data
    if (dataSize > RCM_USER_DATA_LENGTH)
        dataSize = RCM_USER_DATA_LENGTH;
    // copy data into message
    memcpy(request.data, data, dataSize);

    // make sure no pending messages
    rcmIfFlush();

    // send message to RCM
    numBytes = sizeof(request) - RCM_USER_DATA_LENGTH + dataSize;
	rcmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = rcmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the RCM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == RCM_SEND_DATA_CONFIRM)
        {
            // status code
            confirm.status = ntohl(confirm.status);
            // only return OK if status is OK
            if (confirm.status == OK)
            {
                retVal = OK;
            }
        }
    }
    return retVal;

}

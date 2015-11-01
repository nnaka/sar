//_____________________________________________________________________________
//
// Copyright 2011-4 Time Domain Corporation
//
//
// mrm.c
//
//   A collection of functions to communicate with an MRM.
//
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

#ifdef WIN32
#include "winsock2.h"
#else // linux
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "mrmIf.h"
#include "mrm.h"


//_____________________________________________________________________________
//
// #defines 
//_____________________________________________________________________________

#if defined(_WIN32) || defined(_WIN64) 
    #define strcasecmp _stricmp 
    #define strncasecmp _strnicmp 
#endif




//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________


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
// mrmConfigGet - get MRM configuration from radio
//_____________________________________________________________________________

int mrmConfigGet(mrmConfiguration *config)
{
    mrmMsg_GetConfigRequest request;
    mrmMsg_GetConfigConfirm confirm;
    int retVal = ERR, numBytes, i;

    // create request message
	request.msgType = htons(MRM_GET_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_GetConfigConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_GET_CONFIG_CONFIRM)
        {
            // copy config from message to caller's structure
            memcpy(config, &confirm.config, sizeof(*config));
            // Handle byte ordering
            config->nodeId = ntohl(config->nodeId);
            config->scanStartPs = ntohl(config->scanStartPs);
            config->scanEndPs = ntohl(config->scanEndPs);
            config->scanResolutionBins = ntohs(config->scanResolutionBins);
            config->baseIntegrationIndex = ntohs(config->baseIntegrationIndex);
            for (i = 0; i < 4; i++)
                config->segmentNumSamples[i] = ntohs(config->segmentNumSamples[i]);

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
// mrmConfigSet - set MRM configuration in radio
//_____________________________________________________________________________

int mrmConfigSet(mrmConfiguration *config)
{
    mrmMsg_SetConfigRequest request;
    mrmMsg_SetConfigConfirm confirm;
    int retVal = ERR, numBytes, i;

    // create request message
	request.msgType = htons(MRM_SET_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);
    memcpy(&request.config, config, sizeof(*config));

    // Handle byte ordering in config struct
    request.config.nodeId = htonl(config->nodeId);
    request.config.scanStartPs = htonl(config->scanStartPs);
    request.config.scanEndPs = htonl(config->scanEndPs);
    request.config.scanResolutionBins = htons(config->scanResolutionBins);
    request.config.baseIntegrationIndex = htons(config->baseIntegrationIndex);
    for (i = 0; i < 4; i++)
        request.config.segmentNumSamples[i] = htons(config->segmentNumSamples[i]);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if ((confirm.msgType == MRM_SET_CONFIG_CONFIRM ) &&
                (confirm.status == OK))
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmControl - control the MRM scanning process
//_____________________________________________________________________________

int mrmControl(mrm_uint16_t msrScanCount, mrm_uint32_t msrIntervalTimeUs)
{
    mrmMsg_ControlRequest request;
    mrmMsg_ControlConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_CONTROL_REQUEST);
	request.msgId = htons(msgIdCount++);

    // Handle byte ordering
    request.msrScanCount = htons(msrScanCount);
    request.msrIntervalTimeMicroseconds = htonl(msrIntervalTimeUs);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if ((confirm.msgType == MRM_CONTROL_CONFIRM) &&
                (confirm.status == OK))
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmConnect - connect to scan server
//_____________________________________________________________________________
int mrmConnect(int connType, char *addr, mrm_uint32_t *connectionStatusCode)
{
    mrmMsg_ServerConnectRequest request;
    mrmMsg_ServerConnectConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_SERVER_CONNECT_REQUEST);
	request.msgId = htons(msgIdCount++);

    request.connectionType = connType;
    if (connType == MRM_CONNECTION_TYPE_NETWORK)
    {
        // inet_addr converts to network byte order automatically
        request.mrmIPAddr = inet_addr(addr);
        request.mrmPort = 0;
    }
    else
    {
        // if there is a leading "COM", skip over it
        if (strncasecmp(addr, "COM", strlen("COM")) == 0)
            request.mrmSerialPortNum = atoi(addr+3);
        else
            request.mrmSerialPortNum = atoi(addr);
        request.mrmSerialPortNum = htonl(request.mrmSerialPortNum);
    }

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);

        // is this the correct message type and is status good?
        if (confirm.msgType == MRM_SERVER_CONNECT_CONFIRM)
        {
            *connectionStatusCode = ntohl(confirm.connectionStatusCode);
            retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmDisconnect - disconnect from scan server
//_____________________________________________________________________________
int mrmDisconnect(void)
{
    mrmMsg_ServerDisconnectRequest request;
    mrmMsg_ServerDisconnectConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_SERVER_DISCONNECT_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if ((confirm.msgType == MRM_SERVER_DISCONNECT_CONFIRM) &&
                (confirm.status == OK))
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmFilterConfigGet - get MRM filter configuration from radio
//_____________________________________________________________________________

int mrmFilterConfigGet(mrmFilterConfig *config)
{
    mrmMsg_GetFilterConfigRequest request;
    mrmMsg_GetFilterConfigConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_GET_FILTER_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_GetFilterConfigConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_GET_FILTER_CONFIG_CONFIRM)
        {
            // copy config from message to params
            // Handle byte ordering
            config->filterFlags = ntohs(confirm.filterFlags);
            config->motionFilterIndex = confirm.motionFilterIndex;
            config->detectionListThresholdMult = confirm.detectionListThresholdMult;

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
// mrmFilterConfigSet - set MRM filter configuration in radio
//_____________________________________________________________________________

int mrmFilterConfigSet(mrmFilterConfig *config)
{
    mrmMsg_SetFilterConfigRequest request;
    mrmMsg_SetFilterConfigConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_SET_FILTER_CONFIG_REQUEST);
	request.msgId = htons(msgIdCount++);
    request.filterFlags = htons(config->filterFlags);
    request.motionFilterIndex = config->motionFilterIndex;
    request.detectionListThresholdMult = config->detectionListThresholdMult;

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(confirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.status = ntohl(confirm.status);

        // is this the correct message type and is status good?
        if ((confirm.msgType == MRM_SET_FILTER_CONFIG_CONFIRM) &&
                (confirm.status == OK))
            retVal = OK;
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmInfoGet - assemble a scan from the MRM or receive a detection list
//_____________________________________________________________________________

int mrmInfoGet(int timeoutMs, mrmInfo *info)
{
    int done = 0, retVal = ERR, numBytes, index, i;

    mrmIfTimeoutMsSet(timeoutMs);
    info->msg.scanInfo.msgId = 0;
    info->scan = 0;
    while (!done)
    {
        // wait for packet with timeout
        numBytes = mrmIfGetPacket(&info->msg, sizeof(mrmMsg_FullScanInfo));

        // did we get a response from the MRM?
        if (numBytes == sizeof(mrmMsg_FullScanInfo))
        {
            // Handle byte ordering
            info->msg.scanInfo.msgType = ntohs(info->msg.scanInfo.msgType);
            info->msg.scanInfo.msgId = ntohs(info->msg.scanInfo.msgId);

            // is this the correct message type?
            if (info->msg.scanInfo.msgType == MRM_FULL_SCAN_INFO)
            {
                // Handle byte ordering
                info->msg.scanInfo.sourceId = ntohl(info->msg.scanInfo.sourceId);
                info->msg.scanInfo.timestamp = ntohl(info->msg.scanInfo.timestamp);
                info->msg.scanInfo.channelRiseTime = ntohl(info->msg.scanInfo.channelRiseTime);
                info->msg.scanInfo.scanSNRLinear = ntohl(info->msg.scanInfo.scanSNRLinear);
                info->msg.scanInfo.ledIndex = ntohl(info->msg.scanInfo.ledIndex);
                info->msg.scanInfo.lockspotOffset = ntohl(info->msg.scanInfo.lockspotOffset);
                info->msg.scanInfo.scanStartPs = ntohl(info->msg.scanInfo.scanStartPs);
                info->msg.scanInfo.scanStopPs = ntohl(info->msg.scanInfo.scanStopPs);
                info->msg.scanInfo.scanStepBins = ntohs(info->msg.scanInfo.scanStepBins);
                info->msg.scanInfo.numSamplesInMessage = ntohs(info->msg.scanInfo.numSamplesInMessage);
                info->msg.scanInfo.numSamplesTotal = ntohl(info->msg.scanInfo.numSamplesTotal);
                info->msg.scanInfo.messageIndex = ntohs(info->msg.scanInfo.messageIndex);
                info->msg.scanInfo.numMessagesTotal = ntohs(info->msg.scanInfo.numMessagesTotal);

                // if this is the first message, allocate space for the entire waveform
                if (info->msg.scanInfo.messageIndex == 0)
                {
                    if (info->scan)
                        free(info->scan);
                    info->scan = malloc(info->msg.scanInfo.numSamplesTotal * sizeof(mrm_int32_t));
                    if (info->scan == NULL)
                    {
                        printf("Out of memory!\n");
                        return ERR;
                    }
                    index = 0;
                }
                if (info->scan)
                {
                    for (i = 0; i < info->msg.scanInfo.numSamplesInMessage; i++)
                        *(info->scan + index + i) = ntohl(info->msg.scanInfo.scan[i]);
                    index += info->msg.scanInfo.numSamplesInMessage;
                    if (info->msg.scanInfo.messageIndex == info->msg.scanInfo.numMessagesTotal-1)
                    {
                        done = 1;
                        retVal = OK;
                    }
                }

            }
        }
        else if (numBytes == sizeof(mrmMsg_DetectionListInfo))
        {
            // Handle byte ordering
            info->msg.detectionList.msgType = ntohs(info->msg.detectionList.msgType);
            info->msg.detectionList.msgId = ntohs(info->msg.detectionList.msgId);

            // is this the correct message type?
            if (info->msg.detectionList.msgType == MRM_DETECTION_LIST_INFO)
            {
                // Handle byte ordering
                info->msg.detectionList.numDetections = ntohs(info->msg.detectionList.numDetections);
                for (i = 0; i < info->msg.detectionList.numDetections; i++)
                {
                    info->msg.detectionList.detections[i].index =
                            ntohs(info->msg.detectionList.detections[i].index);
                    info->msg.detectionList.detections[i].magnitude =
                            ntohs(info->msg.detectionList.detections[i].magnitude);
                }
                // corner case - might have started assembling a full scan, missed one packet,
                // and then received a detection list message. So free scan if it was allcoated
                if (info->scan)
                    free(info->scan);
                done = 1;
                retVal = OK;
            }
        }
        else
        {
            // timed out waiting, return error
            done = 1;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmOpmodeGet - retrieve mode of operation from radio
//_____________________________________________________________________________

int mrmOpmodeGet(int *mode)
{
    mrmMsg_GetOpmodeRequest request;
    mrmMsg_GetOpmodeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_GET_OPMODE_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_GetOpmodeConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_GET_OPMODE_CONFIRM)
        {
            // Handle byte ordering
            *mode = ntohl(confirm.opMode);

            // status code
            retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmOpmodeSet - change mode of operation of radio
//_____________________________________________________________________________

int mrmOpmodeSet(int mode)
{
    mrmMsg_SetOpmodeRequest request;
    mrmMsg_SetOpmodeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_SET_OPMODE_REQUEST);
	request.msgId = htons(msgIdCount++);
	request.opMode = htonl(mode);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_SetOpmodeConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_SET_OPMODE_CONFIRM)
        {
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
// mrmSleepModeGet - retrieve sleep mode from radio
//_____________________________________________________________________________

int mrmSleepModeGet(int *mode)
{
    mrmMsg_GetSleepModeRequest request;
    mrmMsg_GetSleepModeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_GET_SLEEP_MODE_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_GetSleepModeConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_GET_SLEEP_MODE_CONFIRM)
        {
            // Handle byte ordering
            *mode = ntohl(confirm.sleepMode);

            // status code
            retVal = OK;
        }
    }
    return retVal;
}


//_____________________________________________________________________________
//
// mrmSleepModeSet - change radio's sleep mode
//_____________________________________________________________________________

int mrmSleepModeSet(int mode)
{
    mrmMsg_SetSleepModeRequest request;
    mrmMsg_SetSleepModeConfirm confirm;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_SET_SLEEP_MODE_REQUEST);
	request.msgId = htons(msgIdCount++);
	request.sleepMode = htonl(mode);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(&confirm, sizeof(confirm));

    // did we get a response from the MRM?
    if (numBytes == sizeof(mrmMsg_SetSleepModeConfirm))
    {
        // Handle byte ordering
        confirm.msgType = ntohs(confirm.msgType);
        confirm.msgId = ntohs(confirm.msgId);

        // is this the correct message type?
        if (confirm.msgType == MRM_SET_SLEEP_MODE_CONFIRM)
        {
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
// mrmStatusInfoGet - retrieve MRM status from radio
//_____________________________________________________________________________

int mrmStatusInfoGet(mrmMsg_GetStatusInfoConfirm *confirm)
{
    mrmMsg_GetStatusInfoRequest request;
    int retVal = ERR, numBytes;

    // create request message
	request.msgType = htons(MRM_GET_STATUS_INFO_REQUEST);
	request.msgId = htons(msgIdCount++);

    // make sure no pending messages
    mrmIfFlush();

    // send message to MRM
	mrmIfSendPacket(&request, sizeof(request));

    // wait for response with timeout
    numBytes = mrmIfGetPacket(confirm, sizeof(*confirm));

    // did we get a response from the MRM?
    // see if it's an old style or new style status message
    if (numBytes == sizeof(mrmMsg_GetStatusInfoConfirm) - sizeof(confirm->packageVersionStr))
    {
        // old style message - fix it up
        memcpy(&confirm->status, confirm->packageVersionStr, sizeof(confirm->status));
        strcpy(confirm->packageVersionStr, "N/A");
        // now bump up numBytes so it appears to be the same size as the new message
        numBytes += sizeof(confirm->packageVersionStr);
    }
    if (numBytes == sizeof(mrmMsg_GetStatusInfoConfirm))
    {
        // Handle byte ordering
        confirm->msgType = ntohs(confirm->msgType);
        confirm->msgId = ntohs(confirm->msgId);

        // is this the correct message type?
        if (confirm->msgType == MRM_GET_STATUS_INFO_CONFIRM)
        {
            // Handle byte ordering
            confirm->appVersionBuild = ntohs(confirm->appVersionBuild);
            confirm->uwbKernelVersionBuild = ntohs(confirm->uwbKernelVersionBuild);
            confirm->serialNum = ntohl(confirm->serialNum);
            confirm->temperature = ntohl(confirm->temperature);

            // status code
            //confirm->status = ntohl(confirm->status);
            // only return OK if status is OK
            //if (confirm->status == OK)
                retVal = OK;
        }
    }
    return retVal;
}

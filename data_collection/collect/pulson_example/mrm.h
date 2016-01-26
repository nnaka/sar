//_____________________________________________________________________________
//
// Copyright 2011-4 Time Domain Corporation
//
//
// mrm.h
//
//   Declarations for MRM communications functions.
//
//_____________________________________________________________________________

#ifndef __mrm_h
#define __mrm_h

#ifdef __cplusplus
    extern "C" {
#endif

//_____________________________________________________________________________
//
// #includes
//_____________________________________________________________________________

// pull in message structure declarations
#include "hostInterfaceMRM.h"


//_____________________________________________________________________________
//
// #defines
//_____________________________________________________________________________


//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________


// scans and detections are returned in this struct
typedef struct
{
    union
    {
        mrmMsg_DetectionListInfo    detectionList;
        mrmMsg_FullScanInfo         scanInfo;
    } msg;
    mrm_int32_t *scan; // not valid for detection messages
} mrmInfo;


typedef struct
{
	mrm_uint16_t filterFlags;
	mrm_uint8_t motionFilterIndex;
	mrm_uint8_t detectionListThresholdMult;
} mrmFilterConfig;


//_____________________________________________________________________________
//
//  Function prototypes
//_____________________________________________________________________________


//
//  mrmConnect
//
//  Parameters:  int connType - MRM_CONNECTION_TYPE_NETWORK, _SERIAL, or _USB
//               char *addr - IP address or COM port of radio
//               mrm_uint32_t *connectionStatusCode - returned connection status
//  Return:      OK or ERR
//
//  Only used when connecting to service. Instructs service to connect
//  to the radio with address addr by sending MRM_SERVER_CONNECT_REQUEST
//  message and waits for MRM_SERVER_CONNECT_CONFIRM. The server can connect
//  to radios via IP, serial port, or USB.
//  If confirm message is received, fills in connectionStatusCode and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmConnect(int connType, char *addr, mrm_uint32_t *connectionStatusCode);


//
//  mrmDisconnect
//
//  Parameters:  void
//  Return:      OK or ERR
//
//  Only used when connecting to service. Instructs service to disconnect
//  from the radio by sending MRM_SERVER_DISCONNECT_REQUEST
//  message and waits for MRM_SERVER_DISCONNECT_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmDisconnect(void);


//
//  mrmConfigGet
//
//  Parameters:  mrmConfiguration *config - pointer to structure to hold
//                      configuration
//  Return:      OK or ERR
//
//  Sends MRM_GET_CONFIG_REQUEST to radio and waits for MRM_GET_CONFIG_CONFIRM.
//  If confirm message is received, fills in config and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmConfigGet(mrmConfiguration *config);


//
//  mrmConfigSet
//
//  Parameters:  mrmConfiguration *config - pointer to structure containing
//                      new configuration
//  Return:      OK or ERR
//
//  Sends MRM_SET_CONFIG_REQUEST to radio and waits for MRM_SET_CONFIG_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmConfigSet(mrmConfiguration *config);


//
//  mrmControl
//
//  Parameters:  mrm_uint16_t msrScanCount - 0 - stop MSR
//                                           1 - single shot
//                                           2 to 65534 - run msrScanCount times
//                                           65535 - run forever
//               mrm_uint32_t msrIntervalTimeUs - microseconds between scans
//  Return:      OK or ERR
//
//  Starts or stops MSR scanning. MSR can run for a specific number of scans
//  by specifying a value of from 2 to 65534 for the value of msrScanCount. To
//  run continuously, specify a value of 65535 (0xffff). Specify a value fo 0 to
//  stop a running MSR. The interval in microseconds between the start of
//  successive scans can be specified in the msrIntervalTimeUs parameter. A value
//  of 0 for this parameter will cause MSR to scan as fast as possible.
//  Sends MRM_CONTROL_REQUEST to the radio and waits for MRM_CONTROL_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmControl(mrm_uint16_t msrScanCount, mrm_uint32_t msrIntervalTimeUs);


//
//  mrmFilterConfigGet
//
//  Parameters: mrmFilterConfig *config - pointer to struct to save config
//  Return:      OK or ERR
//
//  Sends MRM_GET_FILTER_CONFIG_REQUEST to radio and waits for MRM_GET_FILTER_CONFIG_CONFIRM.
//  If confirm message is received, fills in struct and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmFilterConfigGet(mrmFilterConfig *config);


//
//  mrmFilterConfigSet
//
//  Parameters: mrmFilterConfig *config - specifies filter config
//  Return:      OK or ERR
//
//  Sends MRM_SET_FILTER_CONFIG_REQUEST to radio and waits for
//  MRM_SET_FILTER_CONFIG_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmFilterConfigSet(mrmFilterConfig *config);


//
//  mrmInfoGet
//
//  Parameters:  int timeoutMs - timeout in ms to wait for info messages
//               mrmInfo *info - pointer to struct to store info
//  Return:      OK or ERR
//
//  Waits up to timeoutMs to receive either scan info or detection list info.
//  Copies either a full scan or a detection list message to the info pointer.
//  Note that if a full scan is received, the message returned in info is only
//  the last full scan message received (full scans are sent in a sequence of
//  full scan info messages). However, the entire scan will be stored in the
//  scan pointer in the info structure. This memory is allocated by this routine
//  and should be freed by the caller.
//  If a message is successfully received, returns OK.
//  If a timeout occurs, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmInfoGet(int timeoutMs, mrmInfo *info);


//
//  mrmOpmodeGet
//
//  Parameters:  int *mode - returned mode
//  Return:      OK or ERR
//
//  Sends MRM_GET_OPMODE_REQUEST to radio and waits for
//  MRM_GET_OPMODE_CONFIRM.
//  If confirm message is received, fills in mode and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmOpmodeGet(int *mode);


//
//  mrmOpmodeSet
//
//  Parameters:  int mode - radio mode to set
//  Return:      OK or ERR
//
//  Sends MRM_SET_OPMODE_REQUEST to radio and waits for
//  MRM_SET_OPMODE_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmOpmodeSet(int mode);


//
//  mrmSleepModeGet
//
//  Parameters:  int *mode - returned mode
//  Return:      OK or ERR
//
//  Sends MRM_GET_SLEEP_MODE_REQUEST to radio and waits for
//  MRM_GET_SLEEP_MODE_CONFIRM.
//  If confirm message is received, fills in mode and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmSleepModeGet(int *mode);


//
//  mrmSleepModeSet
//
//  Parameters:  int mode - sleep mode to set
//  Return:      OK or ERR
//
//  Sends MRM_SET_SLEEP_MODE_REQUEST to radio and waits for
//  MRM_SET_SLEEP_MODE_CONFIRM.
//  If confirm message is received, returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmSleepModeSet(int mode);


//
//  mrmStatusInfoGet
//
//  Parameters:  rcrmMsg_GetStatusInfoConfirm *statusInfo - pointer to structure
//                      to receive status info message
//  Return:      OK or ERR
//
//  Sends MRM_GET_STATUS_INFO_REQUEST to radio and waits for
//  MRM_GET_STATUS_INFO_CONFIRM.
//  If confirm message is received, fills in statusInfo and returns OK.
//  If confirm message is not received, returns ERR.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmStatusInfoGet(mrmMsg_GetStatusInfoConfirm *statusInfo);


#ifdef __cplusplus
    }
#endif


#endif

#ifndef __hostInterfaceMRM_h
#define __hostInterfaceMRM_h

#if !defined(FEATURESET) && !defined(FEATURE_INCLUDE_MRM)
// Looks like we're being included by an app other than the embedded MRM.
// Enable MRM features.

#define FEATURE_INCLUDE_MRM	1
#define FEATURESET	FEATURE_INCLUDE_MRM

#endif

#if (FEATURESET & FEATURE_INCLUDE_MRM)

#include "hostInterfaceCommon.h"


///////////////////////////////////////
//
// Message types
//

// REQUEST messages are sent by the host to the embedded applicaion.
// CONFIRM messages are sent by the embedded application to the host in response to REQUEST messages.
// INFO messages are sent automatically by the embedded application to the host when various events occur.
#define MRM_MSG_TYPE_REQUEST			(0x1000)
#define MRM_MSG_TYPE_CONFIRM			(0x1100)
#define MRM_MSG_TYPE_INFO				(0x1200)


///////////////////////////////////////
//
// Host <-> Embedded conversation messages
//

// Set monostatic radar parameters
#define MRM_SET_CONFIG_REQUEST			(MRM_MSG_TYPE_REQUEST + 1)
#define MRM_SET_CONFIG_CONFIRM			(MRM_MSG_TYPE_CONFIRM + 1)


// Get monostatic radar parameters
#define MRM_GET_CONFIG_REQUEST			(MRM_MSG_TYPE_REQUEST + 2)
#define MRM_GET_CONFIG_CONFIRM			(MRM_MSG_TYPE_CONFIRM + 2)


// Control monostatic radar operation
#define MRM_CONTROL_REQUEST				(MRM_MSG_TYPE_REQUEST + 3)
#define MRM_CONTROL_CONFIRM				(MRM_MSG_TYPE_CONFIRM + 3)

// Only implemented by MRM PC server
// Requests a connection to an actual MRM.
#define MRM_SERVER_CONNECT_REQUEST		(MRM_MSG_TYPE_REQUEST + 4)
#define MRM_SERVER_CONNECT_CONFIRM		(MRM_MSG_TYPE_CONFIRM + 4)

// Only implemented by MRM PC server
// Requests that the MRM server remove its association of the client to the actual MRM.
#define MRM_SERVER_DISCONNECT_REQUEST	(MRM_MSG_TYPE_REQUEST + 5)
#define MRM_SERVER_DISCONNECT_CONFIRM	(MRM_MSG_TYPE_CONFIRM + 5)

// Currently only implemented by the MRM PC server
// Controls scan filtering parameters.
#define MRM_SET_FILTER_CONFIG_REQUEST	(MRM_MSG_TYPE_REQUEST + 6)
#define MRM_SET_FILTER_CONFIG_CONFIRM	(MRM_MSG_TYPE_CONFIRM + 6)

#define MRM_GET_FILTER_CONFIG_REQUEST	(MRM_MSG_TYPE_REQUEST + 7)
#define MRM_GET_FILTER_CONFIG_CONFIRM	(MRM_MSG_TYPE_CONFIRM + 7)

///////////////////////////////////////
//
// Embedded -> Host info messages
//

// Contains a list of detections and magnitudes.
#define MRM_DETECTION_LIST_INFO			(MRM_MSG_TYPE_INFO + 1)
#define MRM_MAX_DETECTION_COUNT			(MRM_MAX_SCAN_SAMPLES)

///////////////////////////////////////
//
// Flags
//

#define MRM_FILTER_RAW				(1 << 0)
#define MRM_FILTER_FASTTIME			(1 << 1)
#define MRM_FILTER_MOTION			(1 << 2)
#define MRM_FILTER_DETECTIONLIST	(1 << 3)

#define MRM_MOTION_FILTER_FIR2		(1)
#define MRM_MOTION_FILTER_FIR3		(2)
#define MRM_MOTION_FILTER_FIR4		(3)
#define MRM_MOTION_FILTER_IIR3		(4)


///////////////////////////////////////
//
// Connection types
//

// Used in MRM_SERVER_CONNECT_REQUEST
#define MRM_CONNECTION_TYPE_NETWORK         (0)
#define MRM_CONNECTION_TYPE_SERIAL          (1)
#define MRM_CONNECTION_TYPE_USB             (2)


///////////////////////////////////////
//
// Status codes
//

// Used in MRM_SERVER_CONNECT_CONFIRM
#define MRM_SERVER_CONNECTIONSTATUSCODE_CONNECTED		0
#define MRM_SERVER_CONNECTIONSTATUSCODE_GENERALERROR	1
#define MRM_SERVER_CONNECTIONSTATUSCODE_MRMALREADYINUSE	2


///////////////////////////////////////
//
// Message struct definitions
//

typedef struct
{
	mrm_uint32_t nodeId;
	
	mrm_int32_t scanStartPs;
	mrm_int32_t scanEndPs;

	mrm_uint16_t scanResolutionBins;

	mrm_uint16_t baseIntegrationIndex;

	mrm_uint16_t segmentNumSamples[4];

	mrm_uint8_t segmentIntMult[4];
	
	// MRM_ANTENNAMODE_TXA_RXB, etc.
	mrm_uint8_t antennaMode;

	// 0 = lowest power, 63 = highest power
	mrm_uint8_t txGain;

	mrm_uint8_t codeChannel;

	mrm_uint8_t persistFlag;
} mrmConfiguration;

typedef struct
{
	// set to MRM_SET_CONFIG_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	mrmConfiguration config;
} mrmMsg_SetConfigRequest;

typedef struct
{
	// set to MRM_SET_CONFIG_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// status code
	mrm_uint32_t status;
} mrmMsg_SetConfigConfirm;

typedef struct
{
	// set to MRM_GET_CONFIG_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_GetConfigRequest;

typedef struct
{
	// set to MRM_GET_CONFIG_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	mrmConfiguration config;

	// milliseconds since radio boot
	mrm_uint32_t timestamp;

	// status code
	mrm_uint32_t status;
} mrmMsg_GetConfigConfirm;

typedef struct
{
	// set to MRM_CONTROL_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// 0 - stop MSR
	// 1 - single shot
	// 2 to 65534 - run msrScanCount times
	// 65535 (0xFFFF) - run forever
	mrm_uint16_t msrScanCount;

	mrm_uint16_t reserved;

	mrm_uint32_t msrIntervalTimeMicroseconds;
} mrmMsg_ControlRequest;

typedef struct
{
	// set to MRM_CONTROL_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// status code
	mrm_uint32_t status;
} mrmMsg_ControlConfirm;


typedef struct
{
    // set to MRM_SERVER_CONNECT_REQUEST
    mrm_uint16_t msgType;
    // identifier to correlate requests with confirms
    mrm_uint16_t msgId;

    union
    {
        // IP address of the actual monostatic radar module.
        // Used when connectionType == MRM_CONNECTION_TYPE_NETWORK.
        mrm_uint32_t mrmIPAddr;

        // Serial port number.
        // Used when connectionType == MRM_CONNECTION_TYPE_SERIAL.
        mrm_uint32_t mrmSerialPortNum;

        // USB port number.
        // Used when connectionType == MRM_CONNECTION_TYPE_USB.
        mrm_uint32_t mrmUSBPortNum;
    };

    // Port of the monostatic radar module - set to 0 to use default
    mrm_uint16_t mrmPort;

    // Type of connection (MRM_CONNECTION_TYPE_*)
    mrm_uint8_t connectionType;

    mrm_uint8_t reserved;
} mrmMsg_ServerConnectRequest;


typedef struct
{
	// set to MRM_SERVER_CONNECT_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	mrm_uint32_t connectionStatusCode;
} mrmMsg_ServerConnectConfirm;

typedef struct
{
	// set to MRM_SERVER_DISCONNECT_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_ServerDisconnectRequest;

typedef struct
{
	// set to MRM_SERVER_DISCONNECT_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// status code
	mrm_uint32_t status;
} mrmMsg_ServerDisconnectConfirm;

typedef struct
{
	// set to MRM_SET_FILTER_CONFIG_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// Specifies which filters to use
	mrm_uint16_t filterFlags;

	mrm_uint8_t motionFilterIndex;

	mrm_uint8_t detectionListThresholdMult;
} mrmMsg_SetFilterConfigRequest;

typedef struct
{
	// set to MRM_SET_FILTER_CONFIG_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// status code
	mrm_uint32_t status;
} mrmMsg_SetFilterConfigConfirm;

typedef struct
{
	// set to MRM_GET_FILTER_CONFIG_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_GetFilterConfigRequest;

typedef struct
{
	// set to MRM_GET_FILTER_CONFIG_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Specifies which filters to use
	mrm_uint16_t filterFlags;

	mrm_uint8_t motionFilterIndex;

	mrm_uint8_t detectionListThresholdMult;

	// status code
	mrm_uint32_t status;
} mrmMsg_GetFilterConfigConfirm;

typedef struct
{
	// set to MRM_DETECTION_LIST_INFO
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	// Number of detections
	mrm_uint16_t numDetections;

	mrm_uint16_t reserved;

	struct
	{
		// Index of scan sample that produced the detection
		mrm_uint16_t index;

		// Relative strength of the detection - railed if necessary.
		mrm_uint16_t magnitude;
	} detections[MRM_MAX_DETECTION_COUNT];
} mrmMsg_DetectionListInfo;

#endif	// #if (FEATURESET & FEATURE_INCLUDE_MRM)
#endif	// #ifdef __hostInterfaceMRM_h

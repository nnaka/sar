///////////////////////////////////////
//
// hostInterfaceRCM.h
//
// Definitions for the interface between a host computer and the embedded RCM.
//
// Copyright (c) 2010-2 Time Domain
//
#ifndef __rcmHostInterfaceRCM_h
#define __rcmHostInterfaceRCM_h

#include "hostInterfaceCommon.h"

// Maximum size of user data
#define RCM_USER_DATA_LENGTH	1024

///////////////////////////////////////
//
// Message types
//

// REQUEST messages are sent by the host to the embedded applicaion.
// CONFIRM messages are sent by the embedded application to the host in response to REQUEST messages.
// INFO messages are sent automatically by the embedded application to the host when various events occur.
#define RCM_MSG_TYPE_REQUEST			(0x0000)
#define RCM_MSG_TYPE_CONFIRM			(0x0100)
#define RCM_MSG_TYPE_INFO				(0x0200)


///////////////////////////////////////
//
// Host <-> Embedded conversation messages
//


// Set configuration
#define RCM_SET_CONFIG_REQUEST			(RCM_MSG_TYPE_REQUEST + 1)
#define RCM_SET_CONFIG_CONFIRM			(RCM_MSG_TYPE_CONFIRM + 1)


// Get configuration
#define RCM_GET_CONFIG_REQUEST			(RCM_MSG_TYPE_REQUEST + 2)
#define RCM_GET_CONFIG_CONFIRM			(RCM_MSG_TYPE_CONFIRM + 2)


// Initiate range request
#define RCM_SEND_RANGE_REQUEST			(RCM_MSG_TYPE_REQUEST + 3)
#define RCM_SEND_RANGE_REQUEST_CONFIRM	(RCM_MSG_TYPE_CONFIRM + 3)


// Send data to another radio
#define RCM_SEND_DATA_REQUEST			(RCM_MSG_TYPE_REQUEST + 4)
#define RCM_SEND_DATA_CONFIRM			(RCM_MSG_TYPE_CONFIRM + 4)


// Set the data the responder will use when responding to a range request.
#define RCM_SET_RESPONSE_DATA_REQUEST	(RCM_MSG_TYPE_REQUEST + 5)
#define RCM_SET_RESPONSE_DATA_CONFIRM	(RCM_MSG_TYPE_CONFIRM + 5)

///////////////////////////////////////
//
// Message struct definitions
//

typedef struct
{
	// ID of the radio in the network
	rcm_uint32_t nodeId;

	rcm_uint16_t integrationIndex;
	rcm_uint8_t antennaMode;
	rcm_uint8_t codeChannel;

	// Two separate electrical delays in picoseconds - one for each antenna
	rcm_int32_t electricalDelayPsA;
	rcm_int32_t electricalDelayPsB;

	rcm_uint16_t flags;

	// 0 = lowest power, 63 = highest power
	rcm_uint8_t txGain;

	// set to non-zero to indicate settings should persist across radio reboots
	rcm_uint8_t persistFlag;
} rcmConfiguration;

///////////////////////////////////////
//
// RCM INFO messages to the host
//

// Results of an attempted range conversation
#define RCM_RANGE_INFO					(RCM_MSG_TYPE_INFO + 1)


// Sent when data was received over the air
#define RCM_DATA_INFO					(RCM_MSG_TYPE_INFO + 2)

// Sent when a waveform scan is completed, if the RCM_SEND_SCANINFO_PKTS flag is set.
// If the complete scan doesn't fit in a single packet, only the portion of the scan centered around the leading edge is included.
#define RCM_SCAN_INFO					(RCM_MSG_TYPE_INFO + 3)


// Definitions of the rcmConfiguration::flags bitfield

// RCM_SEND_SCANINFO_PKTS - when set, SCAN_INFO packets are sent whenever a scan is received.
// Default is disabled/not set.
#define RCM_SEND_SCANINFO_PKTS		(1 << 0)

// RCM_SEND_FULL_SCANINFO_PKTS - when set, FULL_SCAN_INFO packets are sent whenever a scan is received.
// Default is disabled/not set.
#define RCM_SEND_FULL_SCANINFO_PKTS	(1 << 1)

// RCM_DISABLE_FAN - (for Hammerhead Rev B) when set, turns OFF the fan.
// Default is not set/fan enabled.
#define RCM_DISABLE_FAN				(1 << 2)

// RCM_SEND_SCAN_WITH_DATA - when set, data packets cause RX to scan
// Default is not set - data packets do not cause scans.
#define RCM_SEND_SCAN_WITH_DATA		(1 << 3)

// RCM_DISABLE_CRE_RANGE_MSGS - when set, non-precision (CRE + Filtered) range messages are NOT sent to the host.
// Default is not set/send CRE messages.
#define RCM_DISABLE_CRE_RANGE_MSGS	(1 << 4)



// Range status bitmask/codes

// No bits set indicates no failures, i.e. success
#define RCM_RANGE_STATUS_SUCCESS			0

// Timeout is set by itself (since you can't have/know about any other failures in this case)
#define RCM_RANGE_STATUS_TIMEOUT			1

// A ranging conversation failure (besides timeout) may be any combination of these.
#define RCM_RANGE_STATUS_REQUESTERLEDFAIL	(1 << 1)
#define RCM_RANGE_STATUS_RESPONDERLEDFAIL	(1 << 2)
#define RCM_RANGE_STATUS_RANGETOOLONG	    (1 << 5)
#define RCM_RANGE_STATUS_COARSERANGEUPDATE	(1 << 6)

#define RCM_RANGE_TYPE_PRECISION			(1 << 0)
#define RCM_RANGE_TYPE_COARSE				(1 << 1)
#define RCM_RANGE_TYPE_FILTERED				(1 << 2)

// Definitions of the antennaId field in various RCM_***_INFO messages
#define RCM_ANTENNAID_A		RCM_ANTENNAMODE_TXA_RXA
#define RCM_ANTENNAID_B		RCM_ANTENNAMODE_TXB_RXB

///////////////////////////////////////
//
// Host <-> RCM conversation messages
//

typedef struct
{
	// set to RCM_SET_CONFIG_REQUEST
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	rcmConfiguration config;
} rcmMsg_SetConfigRequest;

typedef struct
{
	// set to RCM_SET_CONFIG_CONFIRM
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// status code
	rcm_uint32_t status;
} rcmMsg_SetConfigConfirm;


typedef struct
{
	// set to RCM_GET_CONFIG_REQUEST
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;
} rcmMsg_GetConfigRequest;

typedef struct
{
	// set to RCM_GET_CONFIG_CONFIRM
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	rcmConfiguration config;

	// milliseconds since radio boot
	rcm_uint32_t timestamp;

	// status code
	rcm_uint32_t status;
} rcmMsg_GetConfigConfirm;


typedef struct
{
	// set to RCM_SEND_RANGE_REQUEST
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// ID of the responder radio to send this range request to
	rcm_uint32_t responderId;

	// which antenna(s) to use
	rcm_uint8_t antennaMode;

	rcm_uint8_t reserved;	// alignment

	// number of bytes of data to send (up to 1024)
	rcm_uint16_t dataSize;

	// Data for range request payload.
	// Only the first dataSize bytes need to actually be in the UDP packet.
	rcm_uint8_t data[RCM_USER_DATA_LENGTH];
} rcmMsg_SendRangeRequest;

typedef struct
{
	// set to RCM_SEND_RANGE_REQUEST_CONFIRM
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// status code
	rcm_uint32_t status;
} rcmMsg_SendRangeRequestConfirm;


typedef struct
{
	// set to RCM_SEND_DATA_REQUEST
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// which antenna(s) to use
	rcm_uint8_t antennaMode;

	rcm_uint8_t reserved;	// alignment

	// number of bytes of data to send (up to 1024)
	rcm_uint16_t dataSize;

	// Data for payload.
	// Only the first dataSize bytes need to actually be in the UDP packet.
	rcm_uint8_t data[RCM_USER_DATA_LENGTH];
} rcmMsg_SendDataRequest;

typedef struct
{
	// set to RCM_SEND_DATA_CONFIRM
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// status code
	rcm_uint32_t status;
} rcmMsg_SendDataConfirm;


typedef struct
{
	// set to RCM_SET_RESPONSE_DATA_REQUEST
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	rcm_uint16_t reserved;	// alignment

	// number of bytes of data to send to a requester (up to 1024)
	rcm_uint16_t dataSize;

	// Data for range response payload.
	// Only the first dataSize bytes need to actually be in the UDP packet.
	rcm_uint8_t data[RCM_USER_DATA_LENGTH];
} rcmMsg_SetResponseDataRequest;

typedef struct
{
	// set to RCM_SET_RESPONSE_DATA_CONFIRM
	rcm_uint16_t msgType;
	// identifier to correlate requests with confirms
	rcm_uint16_t msgId;

	// status code
	rcm_uint32_t status;
} rcmMsg_SetResponseDataConfirm;

///////////////////////////////////////
//
// INFO messages to the host
//

typedef struct
{
	// set to RCM_RANGE_INFO
	rcm_uint16_t msgType;
	// identifier to correlate range requests with info messages
	rcm_uint16_t msgId;

	// ID of the responding radio
	rcm_uint32_t responderId;

	// status/error codes for the range conversation
	rcm_uint8_t rangeStatus;
	// Antenna(s) on this radio used in the range conversation
	rcm_uint8_t antennaMode;
	// how long the range conversation took, in milliseconds
	rcm_uint16_t stopwatchTime;

	rcm_uint32_t precisionRangeMm;
	rcm_uint32_t coarseRangeMm;
	rcm_uint32_t filteredRangeMm;
	
	rcm_uint16_t precisionRangeErrEst;
	rcm_uint16_t coarseRangeErrEst;
	rcm_uint16_t filteredRangeErrEst;

	rcm_int16_t filteredRangeVel;
	rcm_uint16_t filteredRangeVelErrEst;

	rcm_uint8_t rangeMeasurementType;
	rcm_uint8_t reserved;
	
	rcm_uint16_t reqLEDFlags;
	rcm_uint16_t respLEDFlags;

	rcm_uint16_t channelRiseTime;
	rcm_uint16_t vPeak;
	
	rcm_int32_t coarseTOFInBins;

	// milliseconds since radio boot at the time of the range conversation
	rcm_uint32_t timestamp;
} rcmMsg_RangeInfo;



typedef struct
{
	// set to RCM_DATA_INFO
	rcm_uint16_t msgType;
	// identifier to correlate range requests with info messages
	rcm_uint16_t msgId;

	// ID of the sending radio
	rcm_uint32_t sourceId;

	// channel quality (LED rise time; lower (faster rise) is better)
	rcm_uint16_t channelRiseTime;
	// Max value in leading edge window
	rcm_uint16_t vPeak;
	// milliseconds since radio boot at the time the data packet was received
	rcm_uint32_t timestamp;

	// Antenna the UWB packet was received on
	rcm_uint8_t antennaId;

	rcm_uint8_t reserved;	// alignment

	// number of bytes of data received, up to 1024
	rcm_uint16_t dataSize;

	// Data from packet payload
	// Only the first dataSize bytes are actually in the UDP packet.
	rcm_uint8_t data[RCM_USER_DATA_LENGTH];
} rcmMsg_DataInfo;

#define RCM_MAX_SCAN_SAMPLES	350

typedef struct
{
	// set to RCM_SCAN_INFO
	rcm_uint16_t msgType;
	// identifier to correlate range requests with info messages
	rcm_uint16_t msgId;

	// ID of the transmitting radio
	rcm_uint32_t sourceId;

	// Antenna the UWB packet was received on
	rcm_uint8_t antennaId;

	rcm_uint8_t reserved;	// alignment
	
	rcm_uint16_t LEDflags;

	// channel quality (LED rise time; lower (faster rise) is better)
	rcm_uint16_t channelRiseTime;
	// Max value in leading edge window
	rcm_uint16_t vPeak;

	// milliseconds since radio boot at the time the data packet was received
	rcm_uint32_t timestamp;

	rcm_int32_t ledIndex;
	rcm_int32_t lockspotOffset;

	// Number of samples provided in the scan
	rcm_uint32_t numSamples;
	// Scan samples. Note that only the first numSamples samples are sent in the UDP packet.
	rcm_int32_t samples[RCM_MAX_SCAN_SAMPLES];
} rcmMsg_ScanInfo;

#endif	// #ifdef __rcmHostInterfaceRCM_h

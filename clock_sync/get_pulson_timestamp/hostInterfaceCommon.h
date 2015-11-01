///////////////////////////////////////
//
// hostInterface.h
//
// Definitions for the interface between a host computer and the embedded RCM.
//
// Copyright (c) 2010 Time Domain
//

#ifndef __hostInterfaceCommon_h
#define __hostInterfaceCommon_h

// Portability

// The section "#ifdef _MSC_VER" was copied from http://msinttypes.googlecode.com/svn/trunk/stdint.h,
// an implementation of stdint.h for Microsoft Visual C/C++ versions earlier than Visual Studio 2010.
#ifdef _MSC_VER

// Visual Studio 6 and Embedded Visual C++ 4 doesn't
// realize that, e.g. char has the same size as __int8
// so we give up on __intX for them.
#if (_MSC_VER < 1300)
typedef signed char       mrm_int8_t;
typedef signed short      mrm_int16_t;
typedef signed int        mrm_int32_t;
typedef unsigned char     mrm_uint8_t;
typedef unsigned short    mrm_uint16_t;
typedef unsigned int      mrm_uint32_t;
#else
typedef signed __int8     mrm_int8_t;
typedef signed __int16    mrm_int16_t;
typedef signed __int32    mrm_int32_t;
typedef unsigned __int8   mrm_uint8_t;
typedef unsigned __int16  mrm_uint16_t;
typedef unsigned __int32  mrm_uint32_t;
#endif

typedef signed __int64       mrm_int64_t;
typedef unsigned __int64     mrm_uint64_t;

#else

typedef	__signed char			mrm_int8_t;
typedef	unsigned char			mrm_uint8_t;
typedef	short					mrm_int16_t;
typedef	unsigned short			mrm_uint16_t;
typedef	int						mrm_int32_t;
typedef	unsigned int			mrm_uint32_t;
typedef	long long				mrm_int64_t;
typedef	unsigned long long		mrm_uint64_t;

#endif


// Socket defines
#define MRM_SOCKET_PORT_NUM  21210


// Internal modes of operation - not all are supported
#define MRM_OPMODE_MRM		1
#define MRM_OPMODE_DEFAULT	MRM_OPMODE_MRM

// P400 sleep modes
#define MRM_SLEEP_MODE_ACTIVE          0
#define MRM_SLEEP_MODE_IDLE            1
#define MRM_SLEEP_MODE_STANDBY_ETH     2 // wakeup via ethernet or serial
#define MRM_SLEEP_MODE_STANDBY_SER     3 // wakeup via serial only
#define MRM_SLEEP_MODE_SLEEP           4 // wakeup via GPIO only

///////////////////////////////////////
//
// Message types
//

// REQUEST messages are sent by the host to the embedded applicaion.
// CONFIRM messages are sent by the embedded application to the host in response to REQUEST messages.
// INFO messages are sent automatically by the embedded application to the host when various events occur.
#define RCRM_MSG_TYPE_REQUEST			(0xF000)
#define RCRM_MSG_TYPE_CONFIRM			(0xF100)
#define RCRM_MSG_TYPE_INFO				(0xF200)


///////////////////////////////////////
//
// Host <-> Embedded conversation messages
//

// Get version and temperature info
#define MRM_GET_STATUS_INFO_REQUEST	(RCRM_MSG_TYPE_REQUEST + 1)
#define MRM_GET_STATUS_INFO_CONFIRM	(RCRM_MSG_TYPE_CONFIRM + 1)


// Reboot P400
#define MRM_REBOOT_REQUEST				(RCRM_MSG_TYPE_REQUEST + 2)
#define MRM_REBOOT_CONFIRM				(RCRM_MSG_TYPE_CONFIRM + 2)

// Set opmode
#define MRM_SET_OPMODE_REQUEST			(RCRM_MSG_TYPE_REQUEST + 3)
#define MRM_SET_OPMODE_CONFIRM			(RCRM_MSG_TYPE_CONFIRM + 3)

// Get opmode
#define MRM_GET_OPMODE_REQUEST			(RCRM_MSG_TYPE_REQUEST + 4)
#define MRM_GET_OPMODE_CONFIRM			(RCRM_MSG_TYPE_CONFIRM + 4)

// Set sleep mode
#define MRM_SET_SLEEP_MODE_REQUEST		(RCRM_MSG_TYPE_REQUEST + 5)
#define MRM_SET_SLEEP_MODE_CONFIRM		(RCRM_MSG_TYPE_CONFIRM + 5)

// Get sleep mode
#define MRM_GET_SLEEP_MODE_REQUEST		(RCRM_MSG_TYPE_REQUEST + 6)
#define MRM_GET_SLEEP_MODE_CONFIRM		(RCRM_MSG_TYPE_CONFIRM + 6)

///////////////////////////////////////
//
// Common INFO messages to the host
//

// Sent when a waveform scan is completed.
// If the complete scan doesn't fit in a single packet, multiple packets are sent which can be combined to form the complete scan.
#define MRM_FULL_SCAN_INFO				(RCRM_MSG_TYPE_INFO + 1)



///////////////////////////////////////
//
// Constants and flags
//


// *_CONFIRM message status codes
#define MRM_CONFIRM_MSG_STATUS_SUCCESS				0
#define MRM_CONFIRM_MSG_STATUS_GENERICFAILURE		1
#define MRM_CONFIRM_MSG_STATUS_WRONGOPMODE			2
#define MRM_CONFIRM_MSG_STATUS_UNSUPPORTEDVALUE	3

// Definitions of the antennaMode field in various messages

// Send RCM_ANTENNAMODE_DEFAULT in a message requiring antennaMode to use the default configured antenna
#define MRM_ANTENNAMODE_DEFAULT		0xff

#define MRM_ANTENNAMODE_TXA_RXA		0
#define MRM_ANTENNAMODE_TXB_RXB		1
#define MRM_ANTENNAMODE_TXA_RXB		2
#define MRM_ANTENNAMODE_TXB_RXA		3


#define MRM_MAX_SCAN_SAMPLES			(350)

// TX gain levels (inverses of the actual attenuation levels)
#define MRM_TXGAIN_MIN		0
#define MRM_TXGAIN_MAX		63


// Scan filtering
#define MRM_SCAN_FILTERING_RAW				(1 << 0)
#define MRM_SCAN_FILTERING_FASTTIME		(1 << 1)
#define MRM_SCAN_FILTERING_MOTION			(1 << 2)


///////////////////////////////////////
//
// Message struct definitions
//



typedef struct
{
	// set to MRM_GET_STATUS_INFO_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_GetStatusInfoRequest;

typedef struct
{
	// set to MRM_GET_STATUS_INFO_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;

	mrm_uint8_t appVersionMajor;
	mrm_uint8_t appVersionMinor;
	mrm_uint16_t appVersionBuild;

	mrm_uint8_t uwbKernelVersionMajor;
	mrm_uint8_t uwbKernelVersionMinor;
	mrm_uint16_t uwbKernelVersionBuild;

	mrm_uint8_t firmwareVersion;
	mrm_uint8_t firmwareYear;
	mrm_uint8_t firmwareMonth;
	mrm_uint8_t firmwareDay;

	mrm_uint32_t serialNum;

	mrm_uint8_t boardRev;
	mrm_uint8_t bitResults;
	mrm_uint8_t featureSet;
	mrm_uint8_t reserved;

	// Divide this by 4 to get temperature in degrees C.
	mrm_int32_t temperature;

    char packageVersionStr[32];

	// status code
	mrm_uint32_t status;
} mrmMsg_GetStatusInfoConfirm;


typedef struct
{
	// set to MRM_REBOOT_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_RebootRequest;

typedef struct
{
	// set to MRM_REBOOT_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_RebootConfirm;

typedef struct
{
	// set to MRM_SET_OPMODE_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Requested operational mode of the P400.
	mrm_uint32_t opMode;
} mrmMsg_SetOpmodeRequest;

typedef struct
{
	// set to MRM_SET_OPMODE_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Opmode of the radio
	mrm_uint32_t opMode;

	mrm_uint32_t status;
} mrmMsg_SetOpmodeConfirm;

typedef struct
{
	// set to MRM_GET_OPMODE_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_GetOpmodeRequest;

typedef struct
{
	// set to MRM_GET_OPMODE_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Current operational mode of the P400.
	mrm_uint32_t opMode;
} mrmMsg_GetOpmodeConfirm;

typedef struct
{
	// set to MRM_SET_SLEEP_MODE_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Requested sleep mode of the P400.
	mrm_uint32_t sleepMode;
} mrmMsg_SetSleepModeRequest;

typedef struct
{
	// set to MRM_SET_SLEEP_MODE_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	mrm_uint32_t status;
} mrmMsg_SetSleepModeConfirm;

typedef struct
{
	// set to MRM_GET_SLEEP_MODE_REQUEST
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
} mrmMsg_GetSleepModeRequest;

typedef struct
{
	// set to MRM_GET_SLEEP_MODE_CONFIRM
	mrm_uint16_t msgType;
	// identifier to correlate requests with confirms
	mrm_uint16_t msgId;
	
	// Current sleep mode of the P400.
	mrm_uint32_t sleepMode;
} mrmMsg_GetSleepModeConfirm;

typedef struct
{
	// set to MRM_READY_INFO
	mrm_uint16_t msgType;
	// identifier to correlate requests with info messages
	mrm_uint16_t msgId;
} mrmMsg_ReadyInfo;

typedef struct
{
	// set to MRM_FULL_SCAN_INFO
	mrm_uint16_t msgType;
	// identifier to correlate range requests with info messages
	mrm_uint16_t msgId;

	// ID of the transmitting radio
	mrm_uint32_t sourceId;

	// Milliseconds since radio boot at the time the scan was completed
	mrm_uint32_t timestamp;

	// channel quality (LED8 rise time; lower (faster rise) is better)
	mrm_uint32_t channelRiseTime;
	// Peak SNR, linear
	mrm_uint32_t scanSNRLinear;

	// These are indices within the assembled scan.
	mrm_int32_t ledIndex;
	mrm_int32_t lockspotOffset;
	
	mrm_int32_t scanStartPs;
	mrm_int32_t scanStopPs;
	
	mrm_uint16_t scanStepBins;
	
	// Raw, fast time, motion, etc.
	mrm_uint8_t scanFiltering;
	
	mrm_uint8_t reserved;	// alignment

	// Antenna the scan was received on
	mrm_uint8_t antennaId;
	
	// The type of operation behind this scan (ranging, BSR, MSR)
	mrm_uint8_t operationMode;
	
	// Number of scan samples in this message
	mrm_uint16_t numSamplesInMessage;

	// Number of samples in the entire scan
	mrm_uint32_t numSamplesTotal;

	// Index of the message in the scan
	mrm_uint16_t messageIndex;

	// Total number of MRM_FULL_SCAN_INFO messages to expect for this particular scan.
	mrm_uint16_t numMessagesTotal;

	// Scan samples.
	// Note that, unlike MRM_SCAN_INFO, this is NOT a variable-sized packet.
	mrm_int32_t scan[MRM_MAX_SCAN_SAMPLES];
} mrmMsg_FullScanInfo;

#endif	// #ifdef __hostInterfaceCommon_h

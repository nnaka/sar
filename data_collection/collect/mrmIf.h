//_____________________________________________________________________________
//
// Copyright 2011-4 Time Domain Corporation
//
//
// mrmIf.h
//
//   Declarations for MRM interface functions.
//
//_____________________________________________________________________________

#ifndef __mrmIf_h
#define __mrmIf_h

#ifdef __cplusplus
    extern "C" {
#endif

//_____________________________________________________________________________
//
// #includes
//_____________________________________________________________________________



//_____________________________________________________________________________
//
// #defines
//_____________________________________________________________________________

#ifndef OK
#define OK 0
#define ERR (-1)
#endif

//_____________________________________________________________________________
//
// typedefs
//_____________________________________________________________________________


typedef enum {mrmIfIp, mrmIfSerial, mrmIfUsb} mrmIfType;

//_____________________________________________________________________________
//
//  Function prototypes
//_____________________________________________________________________________


//
//  mrmIfInit
//
//  Parameters:  mrmIfType ifType - type of connection to MRM
//               char *destAddr - IP address or serial/USB port name
//  Return:      OK or ERR
//
//  Performs initialization necessary for particular type of interface.
//  Returns ERR on failure.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmIfInit(mrmIfType ifType, const char *destAddr);


//
//  mrmIfClose
//
//  Parameters:  void
//  Return:      void
//
//  Closes socket or port to radio.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void mrmIfClose(void);


//
//  mrmIfGetPacket
//
//  Parameters:  void *pkt - pointer to location to receive packet
//               unsigned maxSize - max size of packet to receive
//  Return:      OK or ERR
//
//  Reads from MRM interface until up to maxSize bytes have been received.
//  Returns ERR on read error, otherwise OK.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmIfGetPacket(void *pkt, unsigned maxSize);


//
//  mrmIfSendPacket
//
//  Parameters:  void *pkt - pointer to packet to send
//               unsigned size - size of packet to send
//  Return:      OK or ERR
//
//  Sends packet to MRM interface.
//  Returns ERR on write error, otherwise OK.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
int mrmIfSendPacket(void *pkt, unsigned size);


//
//  mrmIfTimeoutMsSet
//
//  Parameters:  int timeoutMs - timeout in ms
//
//  Return:      void
//
//  Sets timeout in ms for reading data from radio.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void mrmIfTimeoutMsSet(int timeoutMs);


//
//  mrmIfFlush
//
//  Parameters:  void
//
//  Return:      void
//
//  Flushes any unread packets.
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void mrmIfFlush(void);

#ifdef __cplusplus
    }
#endif


#endif

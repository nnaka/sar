
The functions in this folder provide samples of MATLAB code for implementing an
interface to the PulsON 400 (P400) Monostatic Radar Module (MRM). The
MRM Application Programming Interface (API) Specification describes the
complete set of messages supported by MRM.

The interface uses Ethernet User Datagram Protocol (UDP) packets
implemented by instantiating Java classes in MATLAB. It does not require
any special MATLAB toolboxes.

These basic examples include using Java classes to create a UDP socket,
create UDP packets, and send and receive UDP packets. The code also
contains examples of integer data representation, handling byte order,
error handling, and other concepts.

This is a brief description of the basic MRM communication functions included:
sckt_mgr.m - hides java open/close UDP socket code.  Notice the persistent SCKT variable.
str2dat.m - converts a structure to data.
get_cfg_rqst.m - sends a MRM_GET_CONFIG_REQUEST message (see MRM API.)
set_cfg_rqst.m - sends a MRM_SET_CONFIG_REQUEST message (see MRM API.)
parse_msg.m - parses MRM responses into a structure.  Currently only operates on 
  MRM_GET_CONFIG_CONFIRM, MRM_CONTROL_CONFIRM, and MRM_SCAN_INFO structures.
ctl_rqst.m - sends a MRM_CONTROL_REQUEST message to the MRM (see MRM API.)
read_pckt.m - waits TIMEOUT ms for a response from the MRM.  Returns response.

The chng_cfg_xmpl is a good starting example for understanding the code. This
script gets the configuration, changes a parameter value, and sets the
configuration.

The reader is referred to MATLAB documentation regarding the use of Java
classes within MATLAB. Other than the use of Java classes, most of the
code should be familiar to a typical MATLAB programmer.



MRMDemo.m was added as a simple radar application script that makes use of
the MATLAB functions listed above.  
Type 'help MRMDemo' or view MRMDemo.m for more information on this script.

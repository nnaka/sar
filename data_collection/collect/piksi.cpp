// NOTE: This MUST be at the top to use C linkage
extern "C" {
#include <libserialport.h>

#include <libsbp/sbp.h>
#include <libsbp/system.h>
#include <libsbp/navigation.h>
}

#include <stdlib.h>

#include "piksi.h"
#include "debug.h"

using namespace std;

static void check_setup(bool cond, const char *msg) {
    if (cond) {
        LOG("%s", msg);
        exit(EXIT_FAILURE); // TODO: (joshpfosi) Should raise SetupFailure
    }
}

Piksi::Piksi(const string & usb_port) {
    const char *serial_port_name = usb_port.c_str();

    check_setup(sp_get_port_by_name(serial_port_name, &piksi_port) != SP_OK,
            "Cannot find provided serial port");
    check_setup(sp_open(piksi_port, SP_MODE_READ) != SP_OK,
            "Cannot open serial_port for reading");
    check_setup(sp_set_baudrate(piksi_port, 1000000) != SP_OK,
            "Cannot set port baud rate");
    check_setup(sp_set_flowcontrol(piksi_port, SP_FLOWCONTROL_NONE) != SP_OK,
            "Cannot set flow control");
    check_setup(sp_set_bits(piksi_port, 8) != SP_OK,
            "Cannot set data bits");
    check_setup(sp_set_parity(piksi_port, SP_PARITY_NONE) != SP_OK,
            "Cannot set parity");
    check_setup(sp_set_stopbits(piksi_port, 1) != SP_OK,
            "Cannot set stop bits");

    sbp_state_init(&s);
}

// Collects 1 radar pulse
//
// @raises CollectionError
void Piksi::collect() {
    // stub
}

u32 Piksi::piksi_port_read(u8 *buff, u32 n, void *context) {
  (void)context;
  return sp_blocking_read(piksi_port, buff, n, 0);
}

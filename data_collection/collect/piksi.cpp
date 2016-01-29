#include <stdlib.h>
#include <assert.h>

extern "C" {
#include <libserialport.h>
}

#include "piksi.h"
#include "debug.h"

using namespace std;

// ----------------------------------------------------------------------------
// Callback functions to interpret SBP messages.
// Every message ID has a callback associated with it to
// receive and interpret the message payload.
// ----------------------------------------------------------------------------

void sbp_pos_llh_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != nullptr);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->pos_llh = *(msg_pos_llh_t *)msg;
}

void sbp_gps_time_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != nullptr);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->gps_time = *(msg_gps_time_t *)msg;
}

u32 piksi_port_read(u8 *buff, u32 n, void *context) {
    return sp_blocking_read(((Piksi *)context)->piksi_port, buff, n, 0);
}

Piksi::Piksi(const string & usb_port) : callbacks_rcvd(0)  {
    const char *serial_port_name = usb_port.c_str();

    LOG("Constructing Piksi, listening on %s", serial_port_name);
    // open port
    check_or_exit(sp_get_port_by_name(serial_port_name, &piksi_port) != SP_OK,
            "Cannot find provided serial port");
    check_or_exit(sp_open(piksi_port, SP_MODE_READ) != SP_OK,
            "Cannot open serial_port for reading");

    // port setup
    check_or_exit(sp_set_baudrate(piksi_port, 1000000) != SP_OK,
            "Cannot set port baud rate");
    check_or_exit(sp_set_flowcontrol(piksi_port, SP_FLOWCONTROL_NONE) != SP_OK,
            "Cannot set flow control");
    check_or_exit(sp_set_bits(piksi_port, 8) != SP_OK,
            "Cannot set data bits");
    check_or_exit(sp_set_parity(piksi_port, SP_PARITY_NONE) != SP_OK,
            "Cannot set parity");
    check_or_exit(sp_set_stopbits(piksi_port, 1) != SP_OK,
            "Cannot set stop bits");

    sbp_state_init(&s);

    // piksi_port_read requires a reference to `this` in order to access
    // `piksi_port`
    sbp_state_set_io_context(&s, this);

    // Register a node and callback, and 
    // associate them with a specific message ID.
    sbp_register_callback(&s, SBP_MSG_GPS_TIME, &sbp_gps_time_callback,
            this, &gps_time_node);
    sbp_register_callback(&s, SBP_MSG_POS_LLH, &sbp_pos_llh_callback,
            this, &pos_llh_node);

    LOG("%s", "Finished constructing Piksi");
}

Piksi::~Piksi() {
    check_or_exit(sp_close(piksi_port) != SP_OK, "Cannot close port properly");
    sp_free_port(piksi_port);
}

// Collects 1 radar pulse
void Piksi::collect(msg_pos_llh_t &pos, msg_gps_time_t &gps) {
    s8 ret = 0;

    do {
        LOG("Insufficient (%d) callbacks received, expected %d",
                callbacks_rcvd, NUM_CALLBACKS);
        ret = sbp_process(&s, &piksi_port_read);
    } while (ret >= 0 && callbacks_rcvd < NUM_CALLBACKS);

    check_or_exit(ret < 0, "sbp_process error");

    callbacks_rcvd = 0;

    pos = pos_llh;
    gps = gps_time;
}

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

    assert(context != NULL);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->pos_llh = *(msg_pos_llh_t *)msg;
}

void sbp_baseline_ned_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != NULL);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->baseline_ned = *(msg_baseline_ned_t *)msg;
}

void sbp_vel_ned_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != NULL);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->vel_ned = *(msg_vel_ned_t *)msg;
}

void sbp_dops_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != NULL);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->dops = *(msg_dops_t *)msg;
}

void sbp_gps_time_callback(u16 sender_id, u8 len,
        u8 msg[], void *context) {
    (void)sender_id, (void)len;

    assert(context != NULL);

    Piksi *p = (Piksi *)context;

    p->callbacks_rcvd++;
    p->gps_time = *(msg_gps_time_t *)msg;
}

u32 piksi_port_read(u8 *buff, u32 n, void *context) {
    return sp_blocking_read(((Piksi *)context)->piksi_port, buff, n, 0);
}

Piksi::Piksi(const string & usb_port) : NUM_CALLBACKS(5), callbacks_rcvd(0)  {
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
    sbp_register_callback(&s, SBP_MSG_BASELINE_NED, &sbp_baseline_ned_callback,
            this, &baseline_ned_node);
    sbp_register_callback(&s, SBP_MSG_VEL_NED, &sbp_vel_ned_callback,
            this, &vel_ned_node);
    sbp_register_callback(&s, SBP_MSG_DOPS, &sbp_dops_callback,
            this, &dops_node);

    LOG("%s", "Finished constructing Piksi");
}

Piksi::~Piksi() {
    check_or_exit(sp_close(piksi_port) != SP_OK, "Cannot close port properly");
    sp_free_port(piksi_port);
}

// Collects 1 radar pulse
string Piksi::collect() {
    char rj[30];
    char str[1000];
    int str_i;

    s8 ret = 0;

    do {
        LOG("Insufficient (%d) callbacks received, expected %d",
                callbacks_rcvd, NUM_CALLBACKS);
        ret = sbp_process(&s, &piksi_port_read);
    } while (ret >= 0 && callbacks_rcvd < NUM_CALLBACKS);

    check_or_exit(ret < 0, "sbp_process error");

    callbacks_rcvd = 0;

    /* Print data from messages received from Piksi. */
    str_i = 0;
    memset(str, 0, sizeof(str));

    /* Print GPS time. */
    str_i += sprintf(str + str_i, "GPS Time:\n");
    str_i += sprintf(str + str_i, "\tWeek\t\t: %6d\n", (int)gps_time.wn);
    sprintf(rj, "%6.10f", ((float)gps_time.tow + ((float)gps_time.ns/1e6))/1e3);
    str_i += sprintf(str + str_i, "\tSeconds\t: %9s\n", rj);

    /* Print absolute position. */
    str_i += sprintf(str + str_i, "Absolute Position:\n");
    sprintf(rj, "%4.10lf", pos_llh.lat);
    str_i += sprintf(str + str_i, "\tLatitude\t: %17s\n", rj);
    sprintf(rj, "%4.10lf", pos_llh.lon);
    str_i += sprintf(str + str_i, "\tLongitude\t: %17s\n", rj);
    sprintf(rj, "%4.10lf", pos_llh.height);
    str_i += sprintf(str + str_i, "\tHeight\t: %17s\n", rj);
    str_i += sprintf(str + str_i, "\tSatellites\t:     %02d\n", pos_llh.n_sats);

    /* Print NED (North/East/Down) baseline (position vector from base to rover). */
    str_i += sprintf(str + str_i, "Baseline (mm):\n");
    str_i += sprintf(str + str_i, "\tNorth\t\t: %6d\n", (int)baseline_ned.n);
    str_i += sprintf(str + str_i, "\tEast\t\t: %6d\n", (int)baseline_ned.e);
    str_i += sprintf(str + str_i, "\tDown\t\t: %6d\n", (int)baseline_ned.d);

    /* Print NED velocity. */
    str_i += sprintf(str + str_i, "Velocity (mm/s):\n");
    str_i += sprintf(str + str_i, "\tNorth\t\t: %6d\n", (int)vel_ned.n);
    str_i += sprintf(str + str_i, "\tEast\t\t: %6d\n", (int)vel_ned.e);
    str_i += sprintf(str + str_i, "\tDown\t\t: %6d\n", (int)vel_ned.d);

    /* Print Dilution of Precision metrics. */
    str_i += sprintf(str + str_i, "Dilution of Precision:\n");
    sprintf(rj, "%4.2f", ((float)dops.gdop/100));
    str_i += sprintf(str + str_i, "\tGDOP\t\t: %7s\n", rj);
    sprintf(rj, "%4.2f", ((float)dops.hdop/100));
    str_i += sprintf(str + str_i, "\tHDOP\t\t: %7s\n", rj);
    sprintf(rj, "%4.2f", ((float)dops.pdop/100));
    str_i += sprintf(str + str_i, "\tPDOP\t\t: %7s\n", rj);
    sprintf(rj, "%4.2f", ((float)dops.tdop/100));
    str_i += sprintf(str + str_i, "\tTDOP\t\t: %7s\n", rj);
    sprintf(rj, "%4.2f", ((float)dops.vdop/100));
    str_i += sprintf(str + str_i, "\tVDOP\t\t: %7s\n", rj);

    LOG("%s", str);

    return string(str);
}

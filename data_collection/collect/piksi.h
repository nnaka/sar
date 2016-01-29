#pragma once

// piksi.h
// Piksi collects data from a GPS over USB

#include <string>

extern "C" {
#include <libsbp/sbp.h>
#include <libsbp/system.h>
#include <libsbp/navigation.h>
}

class Piksi {
    public:
        Piksi(const std::string &);
        ~Piksi();

        void collect(msg_pos_llh_t &, msg_gps_time_t &);
    private:
        friend void sbp_pos_llh_callback(u16 sender_id, u8 len, u8 msg[], void *context);
        friend void sbp_gps_time_callback(u16 sender_id, u8 len, u8 msg[], void *context);

        friend u32 piksi_port_read(u8 *buff, u32 n, void *context);

        struct sp_port *piksi_port;

        const int NUM_CALLBACKS = 2;
        int callbacks_rcvd;

        sbp_state_t s;

        sbp_msg_callbacks_node_t pos_llh_node;
        sbp_msg_callbacks_node_t gps_time_node;

        /* SBP structs that messages from Piksi will feed. */
        msg_pos_llh_t      pos_llh;
        msg_gps_time_t     gps_time;
};

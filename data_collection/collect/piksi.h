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

        void collect();
    private:
        friend void sbp_pos_llh_callback(u16 sender_id, u8 len, u8 msg[], void *context);
        friend void sbp_baseline_ned_callback(u16 sender_id, u8 len, u8 msg[], void *context);
        friend void sbp_vel_ned_callback(u16 sender_id, u8 len, u8 msg[], void *context);
        friend void sbp_dops_callback(u16 sender_id, u8 len, u8 msg[], void *context);
        friend void sbp_gps_time_callback(u16 sender_id, u8 len, u8 msg[], void *context);

        friend u32 piksi_port_read(u8 *buff, u32 n, void *context);

        struct sp_port *piksi_port;

        const int NUM_CALLBACKS;
        int callbacks_rcvd;

        sbp_state_t s;

        sbp_msg_callbacks_node_t pos_llh_node;
        sbp_msg_callbacks_node_t baseline_ned_node;
        sbp_msg_callbacks_node_t vel_ned_node;
        sbp_msg_callbacks_node_t dops_node;
        sbp_msg_callbacks_node_t gps_time_node;

        /* SBP structs that messages from Piksi will feed. */
        msg_pos_llh_t      pos_llh;
        msg_baseline_ned_t baseline_ned;
        msg_vel_ned_t      vel_ned;
        msg_dops_t         dops;
        msg_gps_time_t     gps_time;
};

#pragma once

// piksi.h
// Piksi collects data from a GPS over USB

#include <string>
#include <libsbp/sbp.h>

class Piksi {
    public:
        Piksi(const std::string &);
        void collect();
    private:
        u32 piksi_port_read(u8 *buff, u32 n, void *context);
        struct sp_port *piksi_port;
        sbp_state_t s;
};

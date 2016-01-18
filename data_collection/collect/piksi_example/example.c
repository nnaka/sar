#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <libserialport.h>

#include <libsbp/sbp.h>
#include <libsbp/system.h>
#include <libsbp/navigation.h>

#define DO_EVERY(n, cmd) do { \
  static u32 do_every_count = 0; \
  if (do_every_count % (n) == 0) { \
    cmd; \
  } \
  do_every_count++; \
} while(0)

char *serial_port_name = NULL;
struct sp_port *piksi_port = NULL;

/* SBP structs that messages from Piksi will feed. */
msg_pos_llh_t      pos_llh;
msg_baseline_ned_t baseline_ned;
msg_vel_ned_t      vel_ned;
msg_dops_t         dops;
msg_gps_time_t     gps_time;

/*
 * SBP callback nodes must be statically allocated. Each message ID / callback
 * pair must have a unique sbp_msg_callbacks_node_t associated with it.
 */
#define NUM_CALLBACKS 5 // ignore heartbeat

static int callbacks_rcvd = 0;
static sbp_msg_callbacks_node_t heartbeat_callback_node;
static sbp_msg_callbacks_node_t pos_llh_node;
static sbp_msg_callbacks_node_t baseline_ned_node;
static sbp_msg_callbacks_node_t vel_ned_node;
static sbp_msg_callbacks_node_t dops_node;
static sbp_msg_callbacks_node_t gps_time_node;

void usage(char *prog_name) {
  fprintf(stderr, "usage: %s [-p serial port]\n", prog_name);
}

void setup_port()
{
  int result;

  result = sp_set_baudrate(piksi_port, 1000000);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot set port baud rate!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_set_flowcontrol(piksi_port, SP_FLOWCONTROL_NONE);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot set flow control!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_set_bits(piksi_port, 8);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot set data bits!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_set_parity(piksi_port, SP_PARITY_NONE);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot set parity!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_set_stopbits(piksi_port, 1);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot set stop bits!\n");
    exit(EXIT_FAILURE);
  }
}

/*
 * Callback functions to interpret SBP messages.
 * Every message ID has a callback associated with it to
 * receive and interpret the message payload.
 */

void heartbeat_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  (void)sender_id, (void)len, (void)msg, (void)context;
  fprintf(stdout, "%s\n", __FUNCTION__);
}

void sbp_pos_llh_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  pos_llh = *(msg_pos_llh_t *)msg;
  ++callbacks_rcvd;
  (void)sender_id, (void)len, (void)context;
}
void sbp_baseline_ned_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  baseline_ned = *(msg_baseline_ned_t *)msg;
  ++callbacks_rcvd;
  (void)sender_id, (void)len, (void)context;
}
void sbp_vel_ned_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  vel_ned = *(msg_vel_ned_t *)msg;
  ++callbacks_rcvd;
  (void)sender_id, (void)len, (void)context;
}
void sbp_dops_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  dops = *(msg_dops_t *)msg;
  ++callbacks_rcvd;
  (void)sender_id, (void)len, (void)context;
}
void sbp_gps_time_callback(u16 sender_id, u8 len, u8 msg[], void *context)
{
  gps_time = *(msg_gps_time_t *)msg;
  ++callbacks_rcvd;
  (void)sender_id, (void)len, (void)context;
}


u32 piksi_port_read(u8 *buff, u32 n, void *context)
{
  (void)context;
  u32 result;

  result = sp_blocking_read(piksi_port, buff, n, 0);

  return result;
}

int main(int argc, char **argv)
{
  int opt;
  int result = 0;

  sbp_state_t s;

  if (argc <= 1) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  while ((opt = getopt(argc, argv, "p:")) != -1) {
    switch (opt) {
      case 'p':
        serial_port_name = (char *)calloc(strlen(optarg) + 1, sizeof(char));
        if (!serial_port_name) {
          fprintf(stderr, "Cannot allocate memory!\n");
          exit(EXIT_FAILURE);
        }
        strcpy(serial_port_name, optarg);
        break;
      case 'h':
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if (!serial_port_name) {
    fprintf(stderr, "Please supply the serial port path where the Piksi is " \
                    "connected!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_get_port_by_name(serial_port_name, &piksi_port);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot find provided serial port!\n");
    exit(EXIT_FAILURE);
  }

  result = sp_open(piksi_port, SP_MODE_READ);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot open %s for reading!\n", serial_port_name);
    exit(EXIT_FAILURE);
  }

  setup_port();

  sbp_state_init(&s);

  /* Register a node and callback, and associate them with a specific message ID. */
  sbp_register_callback(&s, SBP_MSG_HEARTBEAT, &heartbeat_callback,
                        NULL, &heartbeat_callback_node);
  sbp_register_callback(&s, SBP_MSG_GPS_TIME, &sbp_gps_time_callback,
                        NULL, &gps_time_node);
  sbp_register_callback(&s, SBP_MSG_POS_LLH, &sbp_pos_llh_callback,
                        NULL, &pos_llh_node);
  sbp_register_callback(&s, SBP_MSG_BASELINE_NED, &sbp_baseline_ned_callback,
                        NULL, &baseline_ned_node);
  sbp_register_callback(&s, SBP_MSG_VEL_NED, &sbp_vel_ned_callback,
                        NULL, &vel_ned_node);
  sbp_register_callback(&s, SBP_MSG_DOPS, &sbp_dops_callback,
                        NULL, &dops_node);
  char rj[30];
  char str[1000];
  int str_i;

  while(1) {
      sbp_process(&s, &piksi_port_read);

      if (callbacks_rcvd != NUM_CALLBACKS - 1) {
          continue;
      }

      callbacks_rcvd = 0;

      /* Print data from messages received from Piksi. */
      str_i = 0;
      memset(str, 0, sizeof(str));

      str_i += sprintf(str + str_i, "\n\n\n\n");

      /* Print GPS time. */
      str_i += sprintf(str + str_i, "GPS Time:\n");
      str_i += sprintf(str + str_i, "\tWeek\t\t: %6d\n", (int)gps_time.wn);
      sprintf(rj, "%6.10f", ((float)gps_time.tow + ((float)gps_time.ns/1e6))/1e3);
      str_i += sprintf(str + str_i, "\tSeconds\t: %9s\n", rj);
      str_i += sprintf(str + str_i, "\n");

      /* Print absolute position. */
      str_i += sprintf(str + str_i, "Absolute Position:\n");
      sprintf(rj, "%4.10lf", pos_llh.lat);
      str_i += sprintf(str + str_i, "\tLatitude\t: %17s\n", rj);
      sprintf(rj, "%4.10lf", pos_llh.lon);
      str_i += sprintf(str + str_i, "\tLongitude\t: %17s\n", rj);
      sprintf(rj, "%4.10lf", pos_llh.height);
      str_i += sprintf(str + str_i, "\tHeight\t: %17s\n", rj);
      str_i += sprintf(str + str_i, "\tSatellites\t:     %02d\n", pos_llh.n_sats);
      str_i += sprintf(str + str_i, "\n");

      /* Print NED (North/East/Down) baseline (position vector from base to rover). */
      str_i += sprintf(str + str_i, "Baseline (mm):\n");
      str_i += sprintf(str + str_i, "\tNorth\t\t: %6d\n", (int)baseline_ned.n);
      str_i += sprintf(str + str_i, "\tEast\t\t: %6d\n", (int)baseline_ned.e);
      str_i += sprintf(str + str_i, "\tDown\t\t: %6d\n", (int)baseline_ned.d);
      str_i += sprintf(str + str_i, "\n");

      /* Print NED velocity. */
      str_i += sprintf(str + str_i, "Velocity (mm/s):\n");
      str_i += sprintf(str + str_i, "\tNorth\t\t: %6d\n", (int)vel_ned.n);
      str_i += sprintf(str + str_i, "\tEast\t\t: %6d\n", (int)vel_ned.e);
      str_i += sprintf(str + str_i, "\tDown\t\t: %6d\n", (int)vel_ned.d);
      str_i += sprintf(str + str_i, "\n");

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
      str_i += sprintf(str + str_i, "\n");

      printf("\033[H");
      printf("\033[2J");
      printf("%s", str);
  }

  result = sp_close(piksi_port);
  if (result != SP_OK) {
    fprintf(stderr, "Cannot close %s properly!\n", serial_port_name);
  }

  sp_free_port(piksi_port);

  free(serial_port_name);

  return 0;
}

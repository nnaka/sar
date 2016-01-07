# Clock-syncing of the GPS and Pulson by the Hummingboard

## Interface
- Inputs
  - Pulses from GPS and Pulson
- Outputs
  - .csv with mapping of time between GPS and Pulson

## Real-time processing method
- Execute on Hummingboard
- Loop reads GPS once
  - for every f2/f1 reads on Pulson, where f2 and f1 are the frequencies at
  which the GPS and Pulson read
- Error
  - network latency
  - read latency
  - rounding error (f2/f1)
- Mitigation
  - there are so many more pules data points the average will smear the error
- Improvements
  - two threads and two queues


## Linear interpoloation method
- Linearly interpolate points between the GPS points to map one-to-one to the points
of the Pulson

* GPS frequency: 50 Hz

Action Items:

*) Figure out how to setup an ad hoc WiFi network on the Hummingboard
*) Determine the Piksi API call to initiate and stop collection
*) Determine the PulsOn API call to initiate and stop collection
*) Determine the Piksi API call to read data
*) Determine the PulsOn API call to read data
  a) Determine exactly what the format of that data means e.g. what is a "bin"
  and how are pulses represented -- Is amplitude a uint32, or what?
*) Determine PulsOn frequency
*) Determine Piksi frequency

### Program Pseudocode 

```
fopen(csv)

while(waiting_on_signal) {}

while (waiting_for_done_signal) {
  gps_struct = read_from_gps() // blocking

  for (some number of pulses) {
    pulses[i] = read_from_pulson() // blocking
  }

  // average over pulses and associate to gps_struct
  // format and append gps_pulse_struct to CSV file
}

fclose(csv)
```

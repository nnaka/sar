# Synthetic Aperture Radar Image Processing

This repository houses all the source code for SAR image processing and
autofocus algorithms. Refer to each subdirectory's `README` for details.

## `data_collection`

All programs used in real time to collect radar data.

## `clock`

All programs used to synchronize the Hummingboard, Piksi GPS, and PulsON radar
clocks.

## `2d_autofocus`

Similar to `3d_autofocus` but for only a two dimensional aperture. This will
likely be deprecated in favor of a flexible `3d_autofocus` process which can
generate two dimensional apertures.

## `3d_autofocus`

Matlab scripts to implement and generate mocked data for three dimensional
autofocus.

## `gps`

No description.

## `power`

No description.

## `sbp_tutorial`

Tutorial found online which executes basic Piksi GPS interfacing options.

## `sample_pulson_app`

Sample C application which exectutes basic PulsON 410 interfacing options.

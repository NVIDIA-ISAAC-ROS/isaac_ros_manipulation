#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################
set -ex

if [ -z "$ISAAC_ROS_WS" ]; then
  echo "Error: ISAAC_ROS_WS environment variable is not set" >&2
  exit 1
fi

# Check if ENABLE_MANIPULATOR_TESTING is set to 'manual_on_robot'
if [ "$ENABLE_MANIPULATOR_TESTING" != "manual_on_robot" ]; then
  echo "Skipping test: ENABLE_MANIPULATOR_TESTING is not set to 'manual_on_robot'"
  echo "To run this test, set: export ENABLE_MANIPULATOR_TESTING=manual_on_robot"
  exit 0
fi

export OUTPUT_DIR=$ISAAC_ROS_WS/calibration_validation_dataset_after_calibration
export ROS_BAG_OUTPUT_DIR=$OUTPUT_DIR/rosbag

mkdir -p "$OUTPUT_DIR"
mkdir -p "$ROS_BAG_OUTPUT_DIR"

launch_test "$ISAAC_ROS_WS"/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/record_calibration_rosbag_realsense_test.py

export CALIBRATION_VALIDATION_OUTPUT_DIR=$OUTPUT_DIR
export NUM_SQUARE_HEIGHT=9 # THis is for the UR5e target, UR10e target is 12
export NUM_SQUARE_WIDTH=14 # This is for the UR5e target, UR10e target is 14
export LONGER_SIDE_M=0.28  # This is for the UR5e target, UR10e target is 0.36
export MARKER_SIZE_M=0.015 # This is for the UR5e target, UR10e target is 0.022
# Now run the calibration validation test.
python3 -m pytest -s "$ISAAC_ROS_WS"/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/validate_calibration_accuracy_test.py

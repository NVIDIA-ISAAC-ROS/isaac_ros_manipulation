# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_manipulation_bringup)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()
  add_subdirectory(test)

  find_package(launch_testing_ament_cmake REQUIRED)
  # These test QOS setting for the foundation pose workflow in manipulator for object following
  add_launch_test(test/foundationpose_qos_object_following_sim_test.py)
  add_launch_test(test/foundationpose_qos_object_following_realsense_test.py)

  # These test QOS setting for the foundation pose workflow in manipulator for foundation servers
  # These are used for the pick and place workflow
  add_launch_test(test/foundationpose_qos_servers_pick_and_place_sim_test.py)
  add_launch_test(test/foundationpose_qos_servers_pick_and_place_realsense_test.py)

  # These tests test for topic connection correctness, these are heavy tests they run ros2
  # cmd line tools for each node and topic and verify the graph structure
  add_launch_test(test/foundationpose_topic_connections_object_following_test.py TIMEOUT 100)
  add_launch_test(test/foundationpose_topic_connections_pick_and_place_test.py TIMEOUT 100)

  # These test QOS setting for the core nodes in manipulator
  add_launch_test(test/core_nodes_qos_realsense_test.py)

  # Add test to check manipulator cumotion and perception
  add_launch_test(test/flexiv_motion_plan_with_nvblox_test.py TIMEOUT 300)
  add_launch_test(test/flexiv_motion_plan_without_nvblox_test.py TIMEOUT 300)

  # Add test to check robot segmentor correctness
  add_launch_test(test/robot_segmentor_pol_realsense_flexiv_test.py TIMEOUT 100)

  # Internal launch tests that are not part of the release
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/internal_launch_tests.cmake" OPTIONAL)
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch config params test)

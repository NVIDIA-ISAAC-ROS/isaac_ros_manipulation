#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Parse Isaac Sim joint states for the Flexiv Rizon with Grav gripper.

Takes incoming joint states from Isaac Sim and publishes only the 7 arm
joints needed by MoveIt/cuMotion/robot_state_publisher. Filters out
Grav gripper passive and mimic joints.
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

ARM_JOINTS = {
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7',
}


class JointParser(Node):

    def __init__(self):
        super().__init__('joint_parser')
        self.subscription = self.create_subscription(
            JointState,
            'rizon_joint_states',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(
            JointState, 'rizon_parsed_joint_states', 10)

    def listener_callback(self, msg):
        new_msg = JointState()
        new_msg.header = msg.header

        for i, name in enumerate(msg.name):
            if name in ARM_JOINTS:
                new_msg.name.append(name)
                new_msg.position.append(msg.position[i])

        if new_msg.name:
            new_msg.velocity = (
                msg.velocity[:len(new_msg.position)]
                if len(msg.velocity) >= len(new_msg.position)
                else [0.0] * len(new_msg.position))
            new_msg.effort = (
                msg.effort[:len(new_msg.position)]
                if len(msg.effort) >= len(new_msg.position)
                else [0.0] * len(new_msg.position))

        self.publisher.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

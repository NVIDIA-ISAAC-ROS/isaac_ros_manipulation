# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test utils functions for Isaac Manipulator."""

from datetime import datetime
import json
import os
import pathlib
import re
import subprocess
from threading import Thread
import time
import traceback
from typing import Any, Dict, List, Tuple

from action_msgs.msg import GoalStatus
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, TransformStamped
from isaac_ros_manipulation_interfaces.action import (
    AddSegmentationMask, GetObjectPose, GetObjects
)
from isaac_ros_manipulation_interfaces.srv import AddMeshToObject, ClearObjects
from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as RCLPYNode
from rclpy.parameter import Parameter
from rclpy.qos import QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
import tf2_ros
import yaml


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


def get_params_from_config_file_set_in_env(run_test: bool) -> Dict:
    """
    Get parameters from a YAML file set in the ISAAC_ROS_MANIPULATION_TEST_CONFIG env var.

    Args
    ----
        run_test (bool): Whether to run the test, if no, then return empty dict

    Returns
    -------
        Dict: Dictionary of parameters

    Raises
    ------
        FileNotFoundError: If the file is not found
        RuntimeError: If the environment variable is not set

    """
    if not run_test:
        return {}

    env_config_path = os.environ.get('ISAAC_ROS_MANIPULATION_TEST_CONFIG')

    if env_config_path is None:
        raise RuntimeError('ISAAC_ROS_MANIPULATION_TEST_CONFIG environment variable is not set')

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f'File not found at {env_config_path}')

    params = load_yaml_params(env_config_path)

    return params


class QosTestBaseClass(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose integration."""

    _filepath: pathlib.Path = pathlib.Path(os.path.dirname(__file__))
    _excluded_topics: list[str] = []
    _topics_with_sensor_data_qos: list[str] = []
    _check_sensor_data_for_topics_for_only_this_node: dict[str, str] = {}
    _check_sensor_data_for_topics_that_have_publisher: list[str] = []
    _run_test: bool = False

    def test_qos_settings(self):
        """Test that all QoS settings are RELIABLE when object following is enabled."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return
        # Get all available topics
        topics_and_types = self.node.get_topic_names_and_types()
        error_out = False
        should_be_best_effort_but_is_reliable = []
        should_be_reliable_but_is_best_effort = []
        for topic_name, _ in topics_and_types:
            if topic_name in self._excluded_topics:
                continue

            self.node.get_logger().warning(f'Checking topic {topic_name}')
            if topic_name in self._check_sensor_data_for_topics_for_only_this_node:
                qos_profile = self._get_qos_profile(
                    topic_name, self._check_sensor_data_for_topics_for_only_this_node[topic_name])
            elif topic_name in self._check_sensor_data_for_topics_that_have_publisher:
                qos_profile = self._get_qos_profile(
                    topic_name, None, is_publisher=True)
            else:
                qos_profile = self._get_qos_profile(topic_name)
            if topic_name in self._topics_with_sensor_data_qos or \
                    topic_name in self._check_sensor_data_for_topics_that_have_publisher:
                if qos_profile is None:
                    self.node.get_logger().warning(f'Topic {topic_name} has no qos profile')
                    continue
                if qos_profile.reliability != QoSReliabilityPolicy.BEST_EFFORT:
                    should_be_best_effort_but_is_reliable.append(topic_name)
                    error_out = True
                    continue
            else:
                if qos_profile is None:
                    self.node.get_logger().warning(f'Topic {topic_name} has no qos profile')
                    continue
                if qos_profile.reliability != QoSReliabilityPolicy.RELIABLE:
                    should_be_reliable_but_is_best_effort.append(topic_name)
                    error_out = True
                    continue

        if error_out:
            self.fail(f'Error in QOS settings:'
                      f' Should be BEST EFFORT: {should_be_best_effort_but_is_reliable}'
                      f' Should be RELIABLE: {should_be_reliable_but_is_best_effort}')

    def _get_qos_profile(self, topic_name: str, node_name: str = None,
                         is_publisher: bool = False) -> QoSReliabilityPolicy | None:
        """
        Check if a topic is using RELIABLE QoS policy.

        Parameters
        ----------
        topic_name : str
            The name of the topic to check
        node_name : str
            The name of the node to check
        is_publisher : bool
            Whether the topic is a publisher

        Returns
        -------
        QoSReliabilityPolicy | None:
            Returns the QoS policy of the topic, otherwise None

        """
        topic_info = self.node.get_publishers_info_by_topic(topic_name)
        if not topic_info:
            topic_info = self.node.get_subscriptions_info_by_topic(topic_name)
            if not topic_info:
                self.node.get_logger().warning(
                    f'No publishers or subscribers found for topic {topic_name}')
                return None

        qos_profile = None
        for idx, info in enumerate(topic_info):
            if info.node_name != node_name and node_name is not None:
                continue
            if is_publisher:
                if info.endpoint_type != 'PUBLISHER':
                    continue
            self.node.get_logger().warning(f'Topic {topic_name} has info for {idx}: {info}')
            # find out if all topics are using the same qos profile
            if qos_profile is None:
                qos_profile = info.qos_profile
            elif qos_profile.reliability != info.qos_profile.reliability:
                self.node.get_logger().warning(
                    f'Topic {topic_name} has qos profile:'
                    f'{info.qos_profile} but it should be {qos_profile}')
                self.fail(f'Topic {topic_name} has qos profile:'
                          f'{info.qos_profile} but it should be {qos_profile}')

        return qos_profile


class IsaacROSFoundationPoseQosTest(QosTestBaseClass):
    """Test for Isaac ROS Foundation Pose + RT-DETR QOS settings."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  excluded_topics: list[str],
                                  topics_with_sensor_data_qos: list[str],
                                  check_sensor_data_for_topics_for_only_this_node: dict[str, str],
                                  check_sensor_data_for_topics_that_have_publisher: list[str],
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._excluded_topics = excluded_topics
        cls._topics_with_sensor_data_qos = topics_with_sensor_data_qos
        cls._check_sensor_data_for_topics_for_only_this_node = \
            check_sensor_data_for_topics_for_only_this_node
        cls._check_sensor_data_for_topics_that_have_publisher = \
            check_sensor_data_for_topics_that_have_publisher
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class TopicConnectionsGraphTest(IsaacROSBaseTest):
    """Test for graph structure of Foundation Pose and RT-DETR integration."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    _run_test: bool = False

    def test_graph_structure(self):
        """Test that the graph structure matches the expected structure."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return
        # Get all nodes and their relationships
        actual_graph = self._build_actual_graph()
        expected_graph = self._get_expected_graph()

        # Save the graphs for debugging
        self.node.get_logger().info(f'Actual graph: {actual_graph}')

        # Compare the actual graph with the expected graph
        errors = self._compare_graphs(actual_graph, expected_graph)

        # Output errors in a structured way
        if errors:
            self.fail(f'Graph structure validation failed with {len(errors)} errors')

    def _compare_graphs(self, actual: Dict, expected: Dict) -> List[str]:
        """
        Compare the actual and expected graphs and return a list of errors.

        Args:
            actual (Dict): The actual graph
            expected (Dict): The expected graph

        Returns
        -------
            List[str]: A list of errors

        """
        errors = []

        # Check for missing nodes
        missing_nodes = []
        for expected_node in expected.keys():
            node_name = expected_node.split('/')[-1] if '/' in expected_node else expected_node
            if expected_node not in actual:
                missing_nodes.append(
                    f'Missing node: {node_name} - This node was in the expected graph but'
                    f' was not found in the running system')

        if missing_nodes:
            errors.append('\nMISSING NODES:')
            errors.extend(missing_nodes)

        # Check for missing publishers
        missing_publishers = []
        for node_name, node_data in expected.items():
            if node_name not in actual:
                continue

            for expected_pub in node_data.get('publishes', []):
                expected_topic = expected_pub.get('topic')
                expected_subscribers = expected_pub.get('to', [])

                # Find this topic in actual publishers
                found_topic = False
                for actual_pub in actual[node_name].get('publishes', []):
                    if actual_pub.get('topic') == expected_topic:
                        found_topic = True
                        actual_subscribers = actual_pub.get('to', [])

                        # Check missing subscribers
                        for expected_sub in expected_subscribers:
                            if expected_sub not in actual_subscribers:
                                missing_publishers.append(
                                    f'Missing subscriber: {expected_sub} is not subscribed'
                                    f' to {expected_topic} from {node_name}')

                        break

                if not found_topic:
                    missing_publishers.append(
                        f'Missing publication: {node_name} does not publish to {expected_topic}')

        if missing_publishers:
            errors.append('\nMISSING PUBLISHERS:')
            errors.extend(missing_publishers)

        # Check for missing subscribers
        missing_subscribers = []
        for node_name, node_data in expected.items():
            if node_name not in actual:
                continue

            for expected_sub in node_data.get('subscribes', []):
                expected_topic = expected_sub.get('topic')
                expected_publishers = expected_sub.get('from', [])

                # Find this topic in actual subscribers
                found_topic = False
                for actual_sub in actual[node_name].get('subscribes', []):
                    if actual_sub.get('topic') == expected_topic:
                        found_topic = True
                        actual_publishers = actual_sub.get('from', [])

                        # Check missing publishers
                        for expected_pub in expected_publishers:
                            if expected_pub not in actual_publishers:
                                missing_subscribers.append(
                                    f'Missing publisher: {expected_pub} does not publish to'
                                    f' {expected_topic} for {node_name}')

                        break

                if not found_topic:
                    missing_subscribers.append(
                        f'Missing subscription: {node_name} does not subscribe to'
                        f' {expected_topic}')

        if missing_subscribers:
            errors.append('\nMISSING SUBSCRIBERS:')
            errors.extend(missing_subscribers)

        # Log the comparison results
        if errors:
            self.node.get_logger().error('\nGraph Structure Validation Errors:')
            self.node.get_logger().error('==================================================')
            for error in errors:
                self.node.get_logger().error(error)
        else:
            self.node.get_logger().info(
                'Graph structure validation passed. All expected nodes and connections found.')

        return errors

    def _get_node_names_and_namespaces(self) -> List[Tuple[str, str]]:
        """
        Get node names and namespaces using ros2 CLI commands.

        This function provides similar functionality to the ROS 2 API's
        get_node_names_and_namespaces() but uses subprocess with ros2 CLI
        commands to work around limitations in the ROS 2 introspection API
        for composable nodes.

        Returns
        -------
            List[Tuple[str, str]]: List of tuples containing (node_name, namespace)

        """
        self.node.get_logger().info('Getting node names via ros2 CLI...')

        try:
            # Run ros2 node list to get all node names
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True,
                env=os.environ.copy(),  # Use same environment as test
                timeout=10
            )

            # Check if command was successful
            if result.returncode != 0:
                self.node.get_logger().error(f'ros2 node list failed: {result.stderr}')
                return []

            # Parse the node names into the format expected by ROS 2 API
            node_list = [n.strip() for n in result.stdout.splitlines() if n.strip()]
            self.node.get_logger().info(f'Found {len(node_list)} nodes: {node_list}')

            # Parse node names into (node_name, namespace) tuples
            parsed_nodes = []
            for node_path in node_list:
                # Remove leading '/' if present for processing
                clean_path = node_path[1:] if node_path.startswith('/') else node_path

                # Split the node path into parts
                parts = clean_path.split('/')

                if len(parts) == 1:  # Just a node name, no namespace
                    node_name = parts[0]
                    namespace = '/'  # Root namespace
                else:
                    # Last part is the node name, the rest is the namespace
                    node_name = parts[-1]
                    namespace = '/' + '/'.join(parts[:-1])

                parsed_nodes.append((node_name, namespace))

            self.node.get_logger().info(f'Parsed nodes: {parsed_nodes}')
            return parsed_nodes

        except subprocess.TimeoutExpired:
            self.node.get_logger().error('Timeout while getting node names')
            return []
        except Exception as e:
            self.node.get_logger().error(f'Error getting node names: {str(e)}')
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return []

    def _build_actual_graph(self) -> Dict:
        """
        Build a graph of the actual node connections using optimized CLI commands.

        This version is much faster as it gets all topic information at once
        rather than making separate calls for each node and topic.

        Returns
        -------
            Dict: A dictionary representation of the actual node connections

        """
        # Initialize the graph
        graph = {}

        # Get all nodes in the ROS system
        nodes = self._get_node_names_and_namespaces()
        self.node.get_logger().info(f'Found {len(nodes)} nodes through CLI')

        # Create a dictionary of nodes by their fully qualified name AND simple name
        node_by_fqn = {}
        node_by_simple_name = {}  # NEW: Dictionary to look up by simple name

        for node_name, namespace in nodes:
            fqn = f'{namespace}/{node_name}' if namespace != '/' else f'/{node_name}'

            # Skip this test node and ROS internal nodes
            if (node_name == self.node.get_name()
                or node_name in ['_ros2cli_daemon_0']
                or 'launch_ros_' in node_name
                    or 'parameter_bridge' in node_name):
                continue

            node_by_fqn[fqn] = (node_name, namespace)
            node_by_simple_name[node_name] = fqn  # NEW: Map simple name to FQN

            # Initialize node in graph
            graph[fqn] = {
                'publishes': [],
                'subscribes': []
            }

        self.node.get_logger().info(f'Nodes being tracked: {list(graph.keys())}')

        # Get all topics at once without types
        self.node.get_logger().info('Getting all topics at once...')
        try:
            # Run command to get list of all topics WITHOUT types
            topic_result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if topic_result.returncode != 0:
                self.node.get_logger().error(f'ros2 topic list failed: {topic_result.stderr}')
                return graph

            # Parse topics - each line is a topic name
            topics = []
            for line in topic_result.stdout.splitlines():
                topic = line.strip()

                # Skip certain topics
                if topic.startswith('/parameter') or topic.startswith('/rosout'):
                    continue

                topics.append(topic)

            self.node.get_logger().info(f'Found {len(topics)} topics to process')
            self.node.get_logger().info(f'Topics: {topics[:10]}...')  # Log first 10 topics

            # Sample a few topics for detailed investigation
            sample_topics = topics[:5]  # Take first 5 topics
            self.node.get_logger().info(f'Investigating sample topics in detail: {sample_topics}')

            # Get info for all topics
            topic_connection_count = 0
            for topic in topics:
                # Get detailed info for this topic - use ONLY the topic name
                try:
                    self.node.get_logger().debug(f'Getting info for topic: {topic}')
                    # Use a shorter timeout since there are many topics
                    info_result = subprocess.run(
                        ['ros2', 'topic', 'info', '-v', topic],
                        capture_output=True,
                        text=True,
                        timeout=3)

                    if info_result.returncode != 0:
                        self.node.get_logger().warning(
                            f'Could not get info for topic {topic}: {info_result.stderr}')
                        continue

                    # For sample topics, log the entire output
                    if topic in sample_topics:
                        self.node.get_logger().info(
                            f'Sample topic {topic} info output:\n{info_result.stdout}')

                    # Parse publisher and subscriber nodes
                    publishers = []
                    subscribers = []

                    # Split the output into sections
                    info_text = info_result.stdout

                    # Use regex to find all "Node name:" and "Node namespace:" pairs
                    # followed by "Endpoint type:"
                    pattern = (
                        r'Node name: (.+?)\n'
                        r'Node namespace: (.+?)[\s\S]+?'
                        r'Endpoint type: (PUBLISHER|SUBSCRIPTION)'
                    )
                    matches = re.finditer(pattern, info_text)

                    for match in matches:
                        simple_node_name = match.group(1).strip()
                        node_namespace = match.group(2).strip()
                        endpoint_type = match.group(3).strip()

                        # Construct the FQN - same as what we have in our graph
                        fqn = f'{node_namespace}/{simple_node_name}' if node_namespace != '/' \
                              else f'/{simple_node_name}'

                        # Check if this node is in our graph
                        if fqn in graph:
                            if endpoint_type == 'PUBLISHER':
                                publishers.append(fqn)
                            elif endpoint_type == 'SUBSCRIPTION':
                                subscribers.append(fqn)
                        elif simple_node_name in node_by_simple_name:
                            # Try using the mapped FQN if simple name is known
                            mapped_fqn = node_by_simple_name[simple_node_name]
                            if endpoint_type == 'PUBLISHER':
                                publishers.append(mapped_fqn)
                            elif endpoint_type == 'SUBSCRIPTION':
                                subscribers.append(mapped_fqn)

                    # Log what we found for this topic
                    if publishers or subscribers:
                        self.node.get_logger().info(
                            f'Topic {topic} has {len(publishers)} publishers '
                            f'and {len(subscribers)} subscribers')
                        self.node.get_logger().info(f'Publishers: {publishers}')
                        self.node.get_logger().info(f'Subscribers: {subscribers}')

                        # Update the graph with this topic's connections
                        for pub_node in publishers:
                            if pub_node in graph:
                                graph[pub_node]['publishes'].append({
                                    'topic': topic,
                                    'to': subscribers
                                })

                        for sub_node in subscribers:
                            if sub_node in graph:
                                graph[sub_node]['subscribes'].append({
                                    'topic': topic,
                                    'from': publishers
                                })
                        topic_connection_count += 1
                except Exception as e:
                    self.node.get_logger().error(f'Error processing topic {topic}: {str(e)}')
                    self.node.get_logger().error(traceback.format_exc())

        except Exception as e:
            self.node.get_logger().error(f'Error building graph: {str(e)}')
            self.node.get_logger().error(traceback.format_exc())

        self.node.get_logger().info(f'Total topics with connections: {topic_connection_count}')
        self.node.get_logger().info(f'Final graph has {len(graph)} nodes')

        # Check each node's connections
        for node, connections in graph.items():
            pub_count = len(connections['publishes'])
            sub_count = len(connections['subscribes'])
            if pub_count > 0 or sub_count > 0:
                self.node.get_logger().info(
                    f'Node {node} has {pub_count} publishers and {sub_count} subscribers')
            else:
                self.node.get_logger().warning(f'Node {node} has NO connections')

        return graph

    def _get_expected_graph(self) -> Dict:
        """
        Define the expected graph structure.

        Returns
        -------
            Dict: A dictionary representation of the expected node connections

        """
        # Create the expected graph with all nodes and connections
        raise NotImplementedError('This function should be implemented by the test class')


class FoundationPoseTopicConnectionsGraphTest(TopicConnectionsGraphTest):
    """Test for Isaac ROS Foundation Pose + RT-DETR topic connections."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class PoseEstimationPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _monitor_topic_name: str = ''
    _max_latency_time: float = 1.5
    _num_messages_to_receive: int = 10
    _num_warmup_messages: int = 0
    _use_sim_time: bool = False
    _message_type: Any = None
    _run_test: bool = False

    def test_for_pose_estimation_pol(self):
        """Test that verifies that pose estimation are coming in at a certain rate."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return
        self.node.get_logger().info('Starting test for pose estimation POL')
        received_messages = {}

        subs = self.create_logging_subscribers(
            [(self._monitor_topic_name, self._message_type)], received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True)

        try:
            message_times = []
            total_messages_received = 0
            start_time = time.time()
            received_msg = False
            total_needed = self._num_warmup_messages + self._num_messages_to_receive
            while total_messages_received < total_needed:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if time.time() - start_time > 20:
                    break

                if self._monitor_topic_name in received_messages and \
                   len(received_messages[self._monitor_topic_name]) > 0:
                    self.node.get_logger().info('Received message')
                    received_msg = True
                    total_messages_received += 1
                    current_time = self.node.get_clock().now()

                    # Get timestamp from most recent message
                    msg = received_messages[self._monitor_topic_name][-1]
                    msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

                    time_diff = abs((current_time - msg_time).nanoseconds / 1e9)
                    if total_messages_received <= self._num_warmup_messages:
                        self.node.get_logger().info(
                            f'Warmup message {total_messages_received}/'
                            f'{self._num_warmup_messages}. '
                            f'Time difference: {time_diff:.3f} seconds (skipped)')
                    else:
                        message_times.append(time_diff)
                        self.node.get_logger().info(
                            f'Message {len(message_times)} received. '
                            f'Time difference: {time_diff:.3f} seconds')
                    received_messages[self._monitor_topic_name] = []

            # Verify all messages were within acceptable time difference
            self.assertTrue(received_msg, 'No messages received for test duration')
            self.assertGreater(
                len(message_times), 0,
                'No messages received after warmup period')
            max_diff = max(message_times)
            self.assertLess(
                max_diff, self._max_latency_time,
                f'Maximum time difference ({max_diff:.3f}s) exceeded limit of'
                f' {self._max_latency_time}s')

            self.node.get_logger().info(
                f'Successfully received {len(message_times)} messages within'
                f' time constraints (skipped {self._num_warmup_messages} warmup)')

        finally:
            self.node.destroy_subscription(subs)


class IsaacROSFoundationPoseEstimationPolTest(PoseEstimationPolTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  monitor_topic_name: str,
                                  max_latency_time: float,
                                  num_messages_to_receive: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  message_type: Any,
                                  node_startup_delay: float,
                                  num_warmup_messages: int = 0):
        cls._run_test = run_test
        cls._monitor_topic_name = monitor_topic_name
        cls._max_latency_time = max_latency_time
        cls._num_messages_to_receive = num_messages_to_receive
        cls._num_warmup_messages = num_warmup_messages
        cls._use_sim_time = use_sim_time
        cls._message_type = message_type

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace()
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class ObjectDetectionActionClient(RCLPYNode):
    """This class will fire off a action call for object detection."""

    def __init__(self):
        super().__init__('test_object_detection_action_client')
        self._action_client = ActionClient(self, GetObjects, '/get_objects')
        self._goal_handle = None
        self.result_received = False
        self.result_future = None
        self.detected_object_id = None

    def send_goal(self):
        """Send a goal to the object detection action server."""
        self.get_logger().info('Waiting for action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        self.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by the action server.')
        self.result_future = self._goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self.detected_object_id = result.objects[0].object_id
            object_ids = []
            for obj in result.objects:
                object_ids.append(obj.object_id)
            self.get_logger().info(f'Objects detected with ids: {object_ids}')
            self.result_received = True
        else:
            self.get_logger().error(f'Action failed with status: {status}')
            self.result_received = False


class ObjectPoseEstimationActionClient(RCLPYNode):
    """This class will fire off an action call for object pose."""

    def __init__(self):
        super().__init__('test_object_pose_estimation_action_client')
        self._action_client = ActionClient(self, GetObjectPose, '/get_object_pose')
        self._goal_handle = None
        self.result_received = False
        self.result_future = None
        self.result = None

    def send_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.get_logger().info('Waiting for pose estimation action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Pose estimation action server not available after waiting')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.get_logger().info('Pose estimation goal accepted by the action server.')
        self.result_future = self._goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Object pose received: {result.object_pose}')
            self.result_received = True
            self.result = result
        else:
            self.get_logger().error(f'Pose estimation action failed with status: {status}')
            self.result_received = False
            self.result_future = None
            self._goal_handle = None
            self.result = None


class PoseEstimationServersPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_perception_requests: int = 10
    _use_sim_time: bool = False
    _run_test: bool = False
    _spin_thread: Thread = None
    _executor: MultiThreadedExecutor = None
    _detection_client: ObjectDetectionActionClient = None
    _pose_client: ObjectPoseEstimationActionClient = None
    _output_dir: str = ''
    _identifier: str = ''
    _allow_failure_on_first_try: bool = False

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self._detection_client = ObjectDetectionActionClient()
        self._pose_client = ObjectPoseEstimationActionClient()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._detection_client)
        self._executor.add_node(self._pose_client)

        # Start spinning in a separate thread
        self._spin_thread = Thread(target=self._executor.spin)
        self._spin_thread.start()

    def test_for_pose_estimation_servers_pol(self):
        """Test that verifies that pose estimation are coming in at a certain rate."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.setUpClients()

        self.node.get_logger().info('Starting test for pose estimation servers POL')
        failure_count = 0
        time.sleep(10)
        poses = []
        for index in range(self._num_perception_requests):
            time.sleep(1)
            # First detect objects
            self._detection_client.send_goal()

            # Wait for the detection result with a timeout
            start_time = self._detection_client.get_clock().now()
            timed_out = False
            while rclpy.ok() and not self._detection_client.result_received:
                time_node_in_node = self._detection_client.get_clock().now()
                max_timeout = self._max_timeout_time_for_action_call
                if (time_node_in_node - start_time).nanoseconds / 1e9 > max_timeout:
                    if self._allow_failure_on_first_try:
                        self.node.get_logger().warning('Allowing failure on first try')
                        timed_out = True
                        break
                    self.fail('Timeout waiting for object detection result.')
                    break
                rclpy.spin_once(self._detection_client, timeout_sec=0.1)

            if not self._allow_failure_on_first_try:
                self.assertTrue(self._detection_client.result_received,
                                'Object detection action did not return successfully')
            elif timed_out:
                failure_count += 1
                continue

            # Now that we've detected an object, get its pose
            if self._detection_client.detected_object_id is not None:
                self._pose_client.send_goal(self._detection_client.detected_object_id)
                timed_out = False
                # Wait for the pose estimation result with a timeout
                start_time = self._pose_client.get_clock().now()
                while rclpy.ok() and not self._pose_client.result_received:
                    time_node_in_node = self._pose_client.get_clock().now()
                    max_timeout = self._max_timeout_time_for_action_call
                    if (time_node_in_node - start_time).nanoseconds / 1e9 > max_timeout:
                        timed_out = True
                        failure_count += 1
                        break
                    rclpy.spin_once(self._pose_client, timeout_sec=0.1)
                if not self._pose_client.result_received and not timed_out:
                    failure_count += 1
            else:
                pass

            # Save the pose estimation result
            if self._pose_client.result_received:
                pose_estimation_result = self._pose_client.result
                self.node.get_logger().info(f'Pose estimation result: {pose_estimation_result}')
                pose_obj = {
                    'position': {
                        'x': pose_estimation_result.object_pose.position.x,
                        'y': pose_estimation_result.object_pose.position.y,
                        'z': pose_estimation_result.object_pose.position.z,
                    },
                    'orientation': {
                        'x': pose_estimation_result.object_pose.orientation.x,
                        'y': pose_estimation_result.object_pose.orientation.y,
                        'z': pose_estimation_result.object_pose.orientation.z,
                        'w': pose_estimation_result.object_pose.orientation.w,
                    },
                    'frame_id': 'camera_1_color_optical_frame',
                    'pose_name_for_tf': f'{self._identifier}_pose_estimate_{index}',
                }
                poses.append(pose_obj)

            # Reset for next iteration
            self._detection_client.result_received = False
            self._pose_client.result_received = False

        self.node.get_logger().info(f'Failure count: {failure_count}')
        self._executor.shutdown()
        self._spin_thread.join()

        # Save the poses to a file
        with open(os.path.join(self._output_dir, 'poses.json'), 'w') as f:
            json.dump(poses, f)

        if failure_count == 1 and self._allow_failure_on_first_try:
            self.node.get_logger().warning('Allowing failure on first try')
            return

        assert failure_count == 0, f'Pose estimation action failed' \
            f'{failure_count}/{self._num_perception_requests} times'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  num_perception_requests: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  output_dir: str,
                                  identifier: str,
                                  node_startup_delay: float,
                                  allow_failure_on_first_try: bool = False):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._num_perception_requests = num_perception_requests
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._identifier = identifier
        cls._allow_failure_on_first_try = allow_failure_on_first_try

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class IsaacRosManipulationLoadEstimationPolTest(IsaacROSBaseTest):
    """Test for measuring Isaac Manipulator performance under system load."""

    DEFAULT_NAMESPACE = ''

    # Performance monitoring settings
    _test_duration_seconds: int  # Run test for 1 minute
    _output_dir: str  # Default output directory
    _monitor_topic_name: str = ''
    _message_type: Any = None
    _num_warmup_messages: int = 5
    _run_test: bool = False
    _use_sim_time: bool = False
    _robot_segmenter_topic_names: list[str] = []
    _max_latency_in_robot_segmentor_ms: int = 5000

    def test_performance_under_load(self):
        """Test performance metrics of pose estimation under system load."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting performance monitoring test')
        received_messages = {}

        self._all_topic_names = [self._monitor_topic_name] + self._robot_segmenter_topic_names

        # Data collection structures
        latencies = {topic_name: [] for topic_name in self._all_topic_names}
        timestamps = {topic_name: [] for topic_name in self._all_topic_names}
        msg_counts = dict.fromkeys(self._all_topic_names, 0)
        BIG_NUMBER = 1000000
        last_received_time = dict.fromkeys(self._all_topic_names, BIG_NUMBER)

        # Monitoring parameters
        start_time = time.time()
        latest_time = start_time
        sample_interval = 0.1  # seconds

        # Create tuple for subscriptions for robot segmenter topics
        robot_segmenter_subs = [
            (topic_name, Image) for topic_name in self._robot_segmenter_topic_names]

        # Create subscription to monitor messages
        subs = self.create_logging_subscribers(
            [(self._monitor_topic_name, self._message_type)] + robot_segmenter_subs,
            received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)
        failure_due_to_latency = False
        failure_latency_msg = ''
        failure_due_to_hz = False
        failure_hz_msg = ''
        try:
            # Run test for specified duration
            while time.time() - start_time < self._test_duration_seconds:
                # Spin node to process callbacks
                rclpy.spin_once(self.node, timeout_sec=sample_interval)

                current_ros_time = self.node.get_clock().now()
                elapsed = time.time() - start_time

                # Log progress every 5 seconds
                if time.time() - latest_time >= 5.0:
                    self.node.get_logger().info(
                        f'Test running: {elapsed:.1f}s / {self._test_duration_seconds}s, '
                        f'Messages for pose estimation: {msg_counts[self._monitor_topic_name]}')
                    latest_time = time.time()

                # Process received messages
                for monitor_topic_name in self._all_topic_names:
                    if monitor_topic_name in received_messages and \
                       len(received_messages[monitor_topic_name]) > 0:
                        # Get all new messages since last check
                        new_messages = received_messages[monitor_topic_name]

                        for msg in new_messages:
                            # Get timestamp from message
                            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

                            # Calculate latency
                            latency = abs((current_ros_time - msg_time).nanoseconds / 1e9)

                            # Store metrics
                            latencies[monitor_topic_name].append(latency)
                            timestamps[monitor_topic_name].append(elapsed)
                            msg_counts[monitor_topic_name] += 1
                            latency_in_ms = latency * 1000
                            # This one makes sure topic is still ticking
                            if last_received_time[monitor_topic_name] != BIG_NUMBER and \
                               monitor_topic_name in self._robot_segmenter_topic_names:
                                if latency_in_ms > self._max_latency_in_robot_segmentor_ms:
                                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                                        f'{latency_in_ms} ms, which is greater than' \
                                        f'{self._max_latency_in_robot_segmentor_ms} ms'
                                    self.node.get_logger().error(error_msg)
                                    failure_due_to_hz = True
                                    failure_hz_msg = error_msg

                            last_received_time[monitor_topic_name] = time.time()

                            # Skip warmup messages that have high startup latency
                            if monitor_topic_name in self._robot_segmenter_topic_names and \
                               msg_counts[monitor_topic_name] > self._num_warmup_messages:
                                if latency_in_ms > self._max_latency_in_robot_segmentor_ms:
                                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                                        f'{latency_in_ms} ms, which is greater than' \
                                        f'{self._max_latency_in_robot_segmentor_ms} ms'
                                    self.node.get_logger().error(error_msg)
                                    failure_due_to_latency = True
                                    failure_latency_msg = error_msg

                            self.node.get_logger().debug(
                                f'Message {msg_counts[monitor_topic_name]} received.'
                                f'Latency: {latency:.3f} seconds')

                        # Clear processed messages to avoid reprocessing
                        received_messages[monitor_topic_name] = []

            # Generate and save performance visualizations
            self._generate_performance_graphs(latencies, timestamps, msg_counts)

            # Calculate and log statistics
            self._log_performance_statistics(latencies)

            # Also do one final check to make sure topics all ticked
            for monitor_topic_name in self._all_topic_names:
                # Only do it for robot segmenter topics
                if monitor_topic_name not in self._robot_segmenter_topic_names:
                    continue
                if last_received_time[monitor_topic_name] == BIG_NUMBER:
                    self.node.get_logger().error(
                        f'Topic {monitor_topic_name} did not receive any messages')
                    self.fail(f'Topic {monitor_topic_name} did not receive any messages')

                latency_in_ms = (time.time() - last_received_time[monitor_topic_name]) * 1000

                # Have it as 5 times of max latency to make sure topic has not stopped publishing
                if latency_in_ms > 5 * self._max_latency_in_robot_segmentor_ms:
                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                        f'{latency_in_ms} ms, which is greater than' \
                        f'{5 * self._max_latency_in_robot_segmentor_ms} ms'
                    self.node.get_logger().error(error_msg)
                    self.fail(error_msg)

            if failure_due_to_latency:
                self.node.get_logger().error(failure_latency_msg)
                self.fail(failure_latency_msg)

            if failure_due_to_hz:
                self.node.get_logger().error(failure_hz_msg)
                self.fail(failure_hz_msg)

        finally:
            self.node.destroy_subscription(subs)

    def _generate_performance_graphs(self, latencies: dict[str, list[float]],
                                     timestamps: dict[str, list[float]],
                                     msg_counts: dict[str, int]):
        """Generate and save performance visualization graphs."""
        for monitor_topic_name in self._all_topic_names:
            if not latencies[monitor_topic_name]:
                self.node.get_logger().error('No messages received during test')
                return

            # Create output directory if it doesn't exist
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Remove / from monitor_topic_name
            monitor_topic_name_str = monitor_topic_name.replace('/', '_')
            output_file = os.path.join(self._output_dir,
                                       f'performance_{monitor_topic_name_str}_{timestamp_str}.png')

            # Create figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Plot 1: Latency over time
            axs[0].plot(timestamps[monitor_topic_name],
                        latencies[monitor_topic_name], 'b-', linewidth=1)
            axs[0].set_title('Message Latency Over Time')
            axs[0].set_ylabel('Latency (seconds)')
            axs[0].grid(True)

            # Plot 2: Message rate
            # Calculate message rate using rolling window
            if len(timestamps[monitor_topic_name]) > 10:
                window_size = 10
                msg_rates = []
                rate_times = []

                for i in range(window_size, len(timestamps[monitor_topic_name])):
                    time_diff = timestamps[monitor_topic_name][i] - \
                        timestamps[monitor_topic_name][i - window_size]
                    if time_diff > 0:
                        rate = window_size / time_diff
                        msg_rates.append(rate)
                        rate_times.append(timestamps[monitor_topic_name][i])

                axs[1].plot(rate_times, msg_rates, 'g-', linewidth=1)
            else:
                axs[1].text(0.5, 0.5, 'Not enough data for rate calculation',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[1].transAxes)

            axs[1].set_title('Message Rate Over Time')
            axs[1].set_ylabel('Messages per second')
            axs[1].grid(True)

            # Plot 3: Jitter (variation in latency)
            if len(latencies[monitor_topic_name]) > 1:
                # Calculate jitter as the difference between consecutive latencies
                jitter = [abs(latencies[monitor_topic_name][i]
                              - latencies[monitor_topic_name][i - 1])
                          for i in range(1, len(latencies[monitor_topic_name]))]
                jitter_times = timestamps[monitor_topic_name][1:]

                axs[2].plot(jitter_times, jitter, 'r-', linewidth=1)
            else:
                axs[2].text(0.5, 0.5, 'Not enough data for jitter calculation',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[2].transAxes)

            axs[2].set_title('Jitter Over Time')
            axs[2].set_ylabel('Jitter (seconds)')
            axs[2].set_xlabel('Time (seconds)')
            axs[2].grid(True)

            # Add overall title
            plt.suptitle('Performance Metrics', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save figure
            plt.savefig(output_file, dpi=300)
            self.node.get_logger().info(f'Performance graphs saved to: {output_file}')

            # Close figure to free memory
            plt.close(fig)

    def _log_performance_statistics(self, latencies: dict[str, list[float]]):
        """Calculate and log performance statistics."""
        for monitor_topic_name in latencies:
            if not latencies[monitor_topic_name]:
                self.node.get_logger().error('No messages received - cannot calculate statistics')
                return

            # Calculate statistics
            avg_latency = np.mean(latencies[monitor_topic_name])
            min_latency = np.min(latencies[monitor_topic_name])
            max_latency = np.max(latencies[monitor_topic_name])
            std_latency = np.std(latencies[monitor_topic_name])

            if len(latencies[monitor_topic_name]) > 1:
                # Calculate jitter statistics
                jitter = [abs(latencies[monitor_topic_name][i]
                              - latencies[monitor_topic_name][i - 1])
                          for i in range(1, len(latencies[monitor_topic_name]))]
                avg_jitter = np.mean(jitter)
                max_jitter = np.max(jitter)
            else:
                avg_jitter = max_jitter = 0

            # Log statistics
            self.node.get_logger().info(
                f'\nPerformance Statistics Summary:\n'
                f'  Total messages: {len(latencies[monitor_topic_name])}\n'
                f'  Message rate: '
                f'{len(latencies[monitor_topic_name]) / self._test_duration_seconds:.2f}'
                f' msg/sec\n'
                f'  Latency (mean): {avg_latency:.6f} sec\n'
                f'  Latency (min): {min_latency:.6f} sec\n'
                f'  Latency (max): {max_latency:.6f} sec\n'
                f'  Latency (std dev): {std_latency:.6f} sec\n'
                f'  Jitter (avg): {avg_jitter:.6f} sec\n'
                f'  Jitter (max): {max_jitter:.6f} sec\n'
            )


class IsaacRosManipulationLoadEstimationTest(IsaacRosManipulationLoadEstimationPolTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  monitor_topic_name: str,
                                  use_sim_time: bool,
                                  test_duration_seconds: int,
                                  output_dir: str,
                                  nodes: list[Node],
                                  message_type: Any,
                                  robot_segmenter_topic_names: list[str],
                                  node_startup_delay: float,
                                  max_latency_in_robot_segmentor_ms: int,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._monitor_topic_name = monitor_topic_name
        cls._use_sim_time = use_sim_time
        cls._message_type = message_type
        cls._test_duration_seconds = test_duration_seconds
        cls._output_dir = output_dir
        cls._robot_segmenter_topic_names = robot_segmenter_topic_names
        cls._max_latency_in_robot_segmentor_ms = max_latency_in_robot_segmentor_ms
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace()
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class IsaacRosManipulationRobotSegmentorTest(IsaacROSBaseTest):
    """Test to verify robot segmentor correctness check."""

    DEFAULT_NAMESPACE = ''

    # Variables for robot segmentor correctness check
    _run_test: bool = False
    _use_sim_time: bool = False
    _output_dir: str = ''
    _xrdf_file_path: str = ''
    _xrdf_collision_geometry_name: str = ''
    _intrinsics_topic: str = '/front_stereo_camera/left/camera_info'
    _mask_topic: str = '/cumotion/camera_1/robot_mask'
    _joint_states_topic: str = '/isaac_joint_states'
    _depth_topic: str = '/cumotion/camera_1/world_depth'
    _raw_depth_topic: str = '/camera_1/depth_image'
    _links_to_check: list[str] = []
    _num_samples: int = 1000
    _buffer_distance_for_collision_spheres: float = 0.1
    _depth_image_is_float16: bool = False

    def _mark_invalid_points_in_robot_mask(self) -> np.ndarray:
        """Mark invalid points in robot mask."""
        self.invalid_depth_mask = np.zeros_like(self.raw_depth_image)
        invalid_points = np.where(self.raw_depth_image == 0)
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        invalid_points = np.where(self.raw_depth_image == np.inf)
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        invalid_points = np.where(np.isnan(self.raw_depth_image))
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        return self.robot_mask, self.invalid_depth_mask

    def setUpVariables(self):
        """Set up the test environment."""
        self.joint_states = None
        self.tf_buffer = None
        self.tf_listener = None
        self.robot_mask = None
        self.camera_intrinsics = None
        self.robot_collision_spheres = None
        self.received_messages = {}

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # Register logging subscribers
        self.register_logging_subscribers()

        # Load camera intrinsics from YAML file
        self.camera_intrinsics, self.joint_states = self._load_camera_intrinsics_and_joint_states()
        self.robot_mask, self.depth_image, self.raw_depth_image = self._load_mask_and_image()

        # Save depth image
        cv2.imwrite(os.path.join(self._output_dir, 'depth_image.png'), self.depth_image)

        # Save robot mask
        cv2.imwrite(os.path.join(self._output_dir, 'robot_mask.png'), self.robot_mask)
        # Now find all points in raw_depth image that are not valid and mark them 0 in robot_mask
        self.robot_mask, self.invalid_depth_mask = self._mark_invalid_points_in_robot_mask()

        # Save robot mask with invalid points marked
        cv2.imwrite(os.path.join(self._output_dir,
                                 'robot_mask_invalid_points_marked.png'), self.robot_mask)
        cv2.imwrite(os.path.join(self._output_dir,
                                 'invalid_depth_mask.png'), self.invalid_depth_mask)

        # save camera intrinsics and joint states to yaml files
        with open(os.path.join(self._output_dir, 'camera_intrinsics.yaml'), 'w') as f:
            yaml.dump(self.camera_intrinsics, f)
        with open(os.path.join(self._output_dir, 'joint_states.yaml'), 'w') as f:
            yaml.dump(self.joint_states, f)

    def register_logging_subscribers(self):
        """Register logging subscribers for required topics."""
        # Create a logging subscriber for the camera_info topic
        reliable_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._subs = self.create_logging_subscribers(
            [(self._intrinsics_topic, CameraInfo),
             (self._mask_topic, Image),
             (self._joint_states_topic, JointState),
             (self._depth_topic, Image),
             (self._raw_depth_topic, Image)],
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=reliable_qos if self._use_sim_time else rclpy.qos.qos_profile_sensor_data)

    def destroy_logging_subscribers(self):
        """Destroy the logging subscribers."""
        self.node.destroy_subscription(self._subs)

    def _load_camera_intrinsics_and_joint_states(self
                                                 ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """
        Load camera intrinsics and joint states from camera_info and joint_states topics.

        Returns
        -------
            tuple[dict[str, Any], dict[str, Any]]: The camera intrinsics and joint states.

        Raises
        ------
            Exception: If the camera intrinsics and joint states are not received within the
                timeout.

        """
        try:
            # Wait for camera intrinsics to be received (up to 10 seconds)
            self.node.get_logger().info(
                f'Waiting for camera intrinsics on {self._intrinsics_topic}')
            start_time = time.time()

            while time.time() - start_time < self._timeout_seconds_for_service_call:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if (self._intrinsics_topic in self.received_messages
                    and self._joint_states_topic in self.received_messages) and \
                    (len(self.received_messages[self._intrinsics_topic]) > 0
                        and len(self.received_messages[self._joint_states_topic]) > 0):
                    self.node.get_logger().info('Camera intrinsics and joint states received')
                    camera_info_msg = self.received_messages[self._intrinsics_topic][-1]
                    joint_states_msg = self.received_messages[self._joint_states_topic][-1]

                    self.camera_distortion_coefficients = np.array(
                        camera_info_msg.d, dtype=np.float32)
                    self.camera_frame_id = camera_info_msg.header.frame_id
                    # Extract intrinsic parameters from CameraInfo message
                    intrinsics = {
                        'fx': camera_info_msg.k[0],  # Focal length x
                        'fy': camera_info_msg.k[4],  # Focal length y
                        'cx': camera_info_msg.k[2],  # Principal point x
                        'cy': camera_info_msg.k[5],  # Principal point y
                        'width': camera_info_msg.width,
                        'height': camera_info_msg.height,
                        'distortion_model': camera_info_msg.distortion_model,
                        'distortion_coeffs': camera_info_msg.d
                    }

                    self.camera_intrinsics_matrix = np.array(
                        camera_info_msg.k, dtype=np.float32).reshape(3, 3)

                    joint_states = {
                        'position': joint_states_msg.position,
                        'velocity': joint_states_msg.velocity,
                        'effort': joint_states_msg.effort
                    }

                    # Log the intrinsic parameters for debugging
                    self.node.get_logger().info(
                        f'Camera intrinsics: fx={intrinsics["fx"]}, fy={intrinsics["fy"]}, '
                        f'cx={intrinsics["cx"]}, cy={intrinsics["cy"]}'
                    )

                    return intrinsics, joint_states

        except Exception as e:
            self.node.get_logger().error(f'Error loading camera intrinsics: {e}')
            self.fail(f'Error loading camera intrinsics: {e}')

        self.node.get_logger().error('Timed out waiting for camera intrinsics')
        self.fail('Timed out waiting for camera intrinsics')

    def _load_mask_and_image(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the robot mask and depth image from the mask and depth topics.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: The robot mask and depth image.

        Raises
        ------
            Exception: If the mask and depth image are not received within the timeout.

        """
        try:
            # Wait for mask to be received (up to 10 seconds)
            self.node.get_logger().info(
                f'Waiting for mask on {self._mask_topic}')
            start_time = time.time()

            while time.time() - start_time < self._timeout_seconds_for_service_call:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                queue_lengths = {k: len(v) for k, v in self.received_messages.items()}
                self.node.get_logger().info(f'Received message queue lengths: {queue_lengths}')

                if self._mask_topic in self.received_messages and \
                    self._depth_topic in self.received_messages and \
                    self._raw_depth_topic in self.received_messages and \
                    (len(self.received_messages[self._mask_topic]) > 0
                        and len(self.received_messages[self._depth_topic]) > 0
                        and len(self.received_messages[self._raw_depth_topic]) > 0):
                    self.node.get_logger().info('Mask and depth image received')
                    mask_msg = self.received_messages[self._mask_topic][-1]
                    image_msg = self.received_messages[self._depth_topic][-1]

                    if self._depth_image_is_float16:
                        # using realsense hence mask is uint16
                        mask_array = np.frombuffer(mask_msg.data, dtype=np.uint16)
                        mask_array = mask_array.reshape(mask_msg.height, mask_msg.width)
                        mask_array = np.abs(1.0 - (mask_array / 65535.0))
                    else:
                        # using hawk or isaac sim hence mask is float32
                        mask_array = np.frombuffer(mask_msg.data, dtype=np.float32)
                        mask_array = mask_array.reshape(mask_msg.height, mask_msg.width)
                        mask_array = np.abs(1.0 - mask_array)

                    # Convert depth image to numpy array
                    if self._depth_image_is_float16:
                        depth_array = np.frombuffer(
                            image_msg.data, dtype=np.float16).astype(np.float32)
                    else:
                        depth_array = np.frombuffer(image_msg.data, dtype=np.float32)

                    depth_array = depth_array.reshape(image_msg.height, image_msg.width)

                    if self._depth_image_is_float16:
                        raw_depth_array = np.frombuffer(
                            self.received_messages[self._raw_depth_topic][-1].data,
                            dtype=np.float16
                        ).astype(np.float32).reshape(
                            self.received_messages[self._raw_depth_topic][-1].height,
                            self.received_messages[self._raw_depth_topic][-1].width
                        )
                    else:
                        raw_depth_array = np.frombuffer(
                            self.received_messages[self._raw_depth_topic][-1].data,
                            dtype=np.float32
                        ).reshape(
                            self.received_messages[self._raw_depth_topic][-1].height,
                            self.received_messages[self._raw_depth_topic][-1].width
                        )

                    # Create a dictionary to store the robot mask
                    return mask_array, depth_array, raw_depth_array

            self.node.get_logger().error('Timed out waiting for mask and depth image')
            self.fail('Timed out waiting for mask and depth image')

        except Exception as e:
            self.node.get_logger().error(f'Error loading mask and depth image: {e}')
            self.fail(f'Error loading mask and depth image: {e}')

    def _load_robot_collision_spheres(self, xrdf_file_path: str) -> dict[str, np.ndarray]:
        """
        Load robot collision spheres from XRDF file.

        Args
        ----
            xrdf_file_path (str): Path to the XRDF file

        Returns
        -------
            dict[str, np.ndarray]: A dictionary of collision spheres for each link

        Raises
        ------
            Exception: If the XRDF file is not found or the collision spheres are not loaded
                correctly.

        """
        with open(xrdf_file_path) as file_p:
            xrdf_dict = yaml.load(file_p, Loader=yaml.SafeLoader)

        # Extract collision spheres from XRDF
        collision_spheres = {}
        spheres = xrdf_dict['geometry'][self._xrdf_collision_geometry_name]['spheres']
        for link_name, link_data in spheres.items():
            if link_name in self._links_to_check:
                all_pts = []
                for sphere in link_data:
                    pts = self._sample_points_from_sphere(
                        np.array(sphere['center']),
                        sphere['radius'] - sphere['radius'] / 2.0,
                    )
                    all_pts.append(pts)
                collision_spheres[link_name] = np.vstack(all_pts)

                print(link_name, collision_spheres[link_name])
        return collision_spheres

    def _project_point_to_image(self, point_3d: np.ndarray,
                                T: np.ndarray,
                                camera_intrinsics: np.ndarray,
                                camera_distortion_coefficients: np.ndarray) -> np.ndarray:
        """
        Project a 3D point to image coordinates.

        Args
        ----
            point_3d (np.ndarray): 3D point to project
            T (np.ndarray): Transformation matrix
            camera_intrinsics (np.ndarray): Camera intrinsics
            camera_distortion_coefficients (np.ndarray): Camera distortion coefficients

        Returns
        -------
            np.ndarray: Projected 2D points

        Raises
        ------
            Exception: If the projection fails

        """
        tvec = T[:3, 3]
        rvec = R.from_matrix(T[:3, :3]).as_rotvec()
        projected_points, _ = cv2.projectPoints(
            point_3d,
            rvec,
            tvec,
            camera_intrinsics,
            camera_distortion_coefficients
        )
        return projected_points.reshape(-1, 2)

    def _sample_points_from_sphere(self, center: np.ndarray,
                                   radius: float) -> np.ndarray:
        """
        Sample points uniformly from the surface of a sphere.

        Args
        ----
            center: np.ndarray of shape (3,) containing the center of the sphere,
                usually this is [0, 0, 0]
            radius: float containing the radius of the sphere
            num_samples: int containing the number of samples to generate

        Returns
        -------
            np.ndarray of shape (num_samples, 3) containing the sampled points

        """
        # Generate random points from a normal distribution (Gaussian sampling)
        points = np.random.normal(0, 1, (self._num_samples, 3))

        # Normalize to create unit vectors (points on a unit sphere)
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

        # Scale by radius and add center
        return points * radius + center

    def _check_point_in_mask(self, points_2d: np.ndarray, mask: np.ndarray
                             ) -> Tuple[bool, np.ndarray]:
        """
        Check if points are masked out (black) in the segmentation mask.

        Args
        ----
            points_2d: np.ndarray of shape (N, 2) containing pixel coordinates (u, v)
            mask: np.ndarray representing the segmentation mask

        Returns
        -------
            tuple[bool, np.ndarray]: True if any point is black (0) in the mask, False otherwise
                and the coordinates of the points that are black.

        """
        y_coords = points_2d[:, 1].astype(int)
        x_coords = points_2d[:, 0].astype(int)

        valid_points = (y_coords >= 0) & (y_coords < mask.shape[0]) & \
                       (x_coords >= 0) & (x_coords < mask.shape[1])
        y_coords = y_coords[valid_points]
        x_coords = x_coords[valid_points]

        # Get mask values at the specified points
        mask_values = mask[y_coords, x_coords]
        zero_mask_indices = np.where(mask_values == 0)[0]

        # Get the corresponding coordinates where mask value is 0
        zero_coords = np.column_stack((x_coords[zero_mask_indices], y_coords[zero_mask_indices])) \
            if len(zero_mask_indices) > 0 else np.array([])

        return np.any(mask_values == 0), zero_coords

    def _transform_point_to_another_frame(self, points: np.ndarray,
                                          from_frame: str,
                                          to_frame: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a point or array of points from one frame to another.

        Args
        ----
            points (np.ndarray): Array of points with shape (N, 3)
            from_frame (str): Source frame
            to_frame (str): Target frame

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: Transformed point(s) in the target frame

        Raises
        ------
            Exception: If the transformation fails

        """
        try:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    # Spin a few times to process messages
                    for _ in range(5):
                        rclpy.spin_once(self.node, timeout_sec=0.1)

                    transform_stamped = self.tf_buffer.lookup_transform(
                        to_frame,
                        from_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        self.node.get_logger().warning(
                            f'Transform lookup attempt {attempt+1}/{max_attempts} failed, '
                            f'retrying...'
                        )
                    else:
                        self.node.get_logger().error(
                            f'All {max_attempts} transform lookup attempts failed: {e}'
                        )
                        self.fail(f'All {max_attempts} transform lookup attempts failed: {e}')

            # Extract translation and rotation from transform
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation

            # Convert to numpy matrix
            translation_array = np.array([translation.x, translation.y, translation.z])
            rotation_array = np.array([rotation.x, rotation.y, rotation.z, rotation.w])

            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(rotation_array).as_matrix()

            # Create transformation matrix (4x4)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation_array

            # Convert to homogeneous coordinates
            homogeneous_points = np.ones((points.shape[0], 4), dtype=np.float32)
            homogeneous_points[:, :3] = points

            # Apply transformation
            transformed_homogeneous = np.dot(homogeneous_points, transform_matrix.T)

            # Convert back to 3D coordinates
            transformed_points = transformed_homogeneous[:, :3]

            return transformed_points, transform_matrix

        except Exception as e:
            self.node.get_logger().error(
                f'Error transforming point from {from_frame} to {to_frame}: {str(e)}')
            self.fail(f'Error transforming point from {from_frame} to {to_frame}: {str(e)}')

    def visualize_sphere_points_in_image(self, points_2d, mask_height, mask_width, output_dir,
                                         output_filename='mask_image_with_pts.png',
                                         side_length=20, color=(0, 255, 0),
                                         add_extension_to_pixels=True) -> np.ndarray:
        """
        Visualize projected sphere points in an image by drawing squares around them.

        Args
        ----
            points_2d (np.ndarray): Array of 2D points with shape (N, 2)
            mask_height (int): Height of the mask image
            mask_width (int): Width of the mask image
            output_dir (str): Directory to save the output image
            output_filename (str, optional): Filename for the output image.
                Defaults to 'mask_image_with_pts.png'.
            side_length (int, optional): Side length of the square around each point.
                Defaults to 20.
            color (tuple, optional): Color for the squares in BGR format. Defaults to (0, 255, 0)
                (green).
            add_extension_to_pixels (bool, optional): Whether to add an extension to the pixels
                of the square. Defaults to True.

        Returns
        -------
            np.ndarray: The image with visualized points

        """
        # Make a copy of the image to avoid modifying the original
        vis_image = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        side_length = side_length // 2
        for i in range(points_2d.shape[0]):
            # Get the point coordinates
            y, x = int(points_2d[i, 1]), int(points_2d[i, 0])

            if y < 0 or y >= vis_image.shape[0] or x < 0 or x >= vis_image.shape[1]:
                continue

            vis_image[y, x] = color

            if add_extension_to_pixels:
                # Draw a square around the point
                cv2.rectangle(
                    vis_image,
                    (x - side_length, y - side_length), (x + side_length, y + side_length),
                    color, 2)

        # Save the image
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, vis_image)
        self.node.get_logger().info(f'Saved visualization image to {output_path}')

        return vis_image

    def test_correctness_of_robot_segmentor(self):
        """
        Test correctness of robot segmentor.

        This function tests the correctness of the robot segmentor by projecting the collision
        spheres onto the image and checking if they are in the segmentation mask.

        Raises
        ------
            Exception: If the robot segmentor is not correct

        """
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting robot segmentor correctness check')
        self.setUpVariables()

        # Construct collision sphere points in wrist_3_link frame
        pts_for_link = np.array([[0, 0, 0]], dtype=np.float32)

        # Load collision spheres from XRDF file
        collision_spheres = self._load_robot_collision_spheres(self._xrdf_file_path)
        all_pts_in_2d = []
        fail_on_exit = False
        # Iterate through each link and its collision spheres
        for link in self._links_to_check:
            pts_for_link = collision_spheres[link]

            # Transform points to camera frame
            _, transform_matrix = self._transform_point_to_another_frame(
                pts_for_link, link, self.camera_frame_id)

            # Project points to image
            points_2d = self._project_point_to_image(pts_for_link,
                                                     transform_matrix,
                                                     self.camera_intrinsics_matrix,
                                                     self.camera_distortion_coefficients)

            all_pts_in_2d.append(points_2d)

            # Check if points are in the segmentation mask
            any_point_outside_mask, zero_coords = self._check_point_in_mask(
                points_2d.reshape(-1, 2), self.robot_mask)
            if any_point_outside_mask:
                # Only fail on exit if points outside the mask are a certain percentage
                if zero_coords.shape[0] > 0.1 * points_2d.shape[0]:
                    fail_on_exit = True

                self.visualize_sphere_points_in_image(
                    zero_coords,
                    self.robot_mask.shape[0],
                    self.robot_mask.shape[1],
                    self._output_dir,
                    color=(0, 0, 255),
                    add_extension_to_pixels=False,
                    output_filename=f'{link}_points_not_in_mask.png'
                )
                self.node.get_logger().error(
                    f'{zero_coords.shape[0]}/{points_2d.shape[0]} points are not in the '
                    f'segmentation mask for {link}')

        # Visualize the projected points on the mask image
        self.visualize_sphere_points_in_image(
            np.vstack(all_pts_in_2d),
            self.robot_mask.shape[0],
            self.robot_mask.shape[1],
            self._output_dir,
            color=(0, 255, 0),
            output_filename='projected_points_on_mask.png',
            add_extension_to_pixels=False
        )

        self.destroy_logging_subscribers()

        if fail_on_exit:
            self.fail('Points are not in the segmentation mask')


class IsaacRosManipulationRobotSegmentorCorrectnessCheckTest(
        IsaacRosManipulationRobotSegmentorTest):
    """Test to verify robot segmentor correctness check."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  links_to_check: list[str],
                                  output_dir: str,
                                  xrdf_file_path: str,
                                  intrinsics_topic: str,
                                  mask_topic: str,
                                  joint_states_topic: str,
                                  depth_topic: str,
                                  raw_depth_topic: str,
                                  num_samples: int,
                                  buffer_distance_for_collision_spheres: float,
                                  depth_image_is_float16: bool,
                                  xrdf_collision_geometry_name: str,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        cls._xrdf_file_path = xrdf_file_path
        cls._intrinsics_topic = intrinsics_topic
        cls._mask_topic = mask_topic
        cls._joint_states_topic = joint_states_topic
        cls._depth_topic = depth_topic
        cls._raw_depth_topic = raw_depth_topic
        cls._links_to_check = links_to_check
        cls._num_samples = num_samples
        cls._buffer_distance_for_collision_spheres = buffer_distance_for_collision_spheres
        cls._depth_image_is_float16 = depth_image_is_float16
        cls._xrdf_collision_geometry_name = xrdf_collision_geometry_name
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace()
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class ManipulatorRobotSegmentorPolTest(IsaacROSBaseTest):
    """Test to verify robot segmentor is running at required hz."""

    DEFAULT_NAMESPACE = ''

    _run_test: bool = False
    _use_sim_time: bool = False
    _output_dir: str = ''
    _timeout_seconds_for_service_call: int = 10.0
    _intrinsics_topics: list[str] = []
    _input_topics: list[str] = []
    _output_topics: list[str] = []
    _time_to_run: int = 60
    _max_latency_ms: int = 500

    def register_logging_subscribers(self):
        """Register logging subscribers for required topics."""
        # Create a logging subscriber for the camera_info topic
        self.received_messages = {}
        list_of_subscriptions = []
        for intrinsics_topic in self._intrinsics_topics:
            list_of_subscriptions.append((intrinsics_topic, CameraInfo))
        for input_topic in self._input_topics:
            list_of_subscriptions.append((input_topic, Image))

        for output_topic in self._output_topics:
            list_of_subscriptions.append((output_topic, Image))

        self._subs = self.create_logging_subscribers(
            list_of_subscriptions,
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

    def test_robot_segmentor_pol(self):
        """Test to verify robot segmentor is running at required hz."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting robot segmentor correctness check')
        self.register_logging_subscribers()

        self._topics_to_check = []
        self._topics_to_check += self._intrinsics_topics
        self._topics_to_check += self._input_topics
        self._topics_to_check += self._output_topics
        self._data_dict = {}
        # Iterate through the topics and check if the data is being published at the required hz
        start_time = time.time()
        failure_latency_msg = ''
        if_failure_latency = False
        while True:

            rclpy.spin_once(self.node, timeout_sec=0.1)

            for topic in self._topics_to_check:
                self.node.get_logger().info(f'Checking topic: {topic}')
                self.node.get_logger().info(f'Topic type: {type(self.received_messages[topic])}')
                self.node.get_logger().info(f'Number of messages: '
                                            f'{len(self.received_messages[topic])}')
                # Store data to find out hz of each topic and number of messages received
                if topic not in self._data_dict:
                    self._data_dict[topic] = {
                        'timestamp_ns': [],
                        'num_messages': 0,
                        'latency_ms': []
                    }

                if len(self.received_messages[topic]) == 0:
                    continue

                # Calculate hz of each topic
                current_timestamp = self.received_messages[topic][-1].header.stamp
                # Get timestamp in nanoseconds
                current_timestamp_ns = current_timestamp.sec * 1e9 + current_timestamp.nanosec
                self._data_dict[topic]['timestamp_ns'].append(current_timestamp_ns)
                self._data_dict[topic]['num_messages'] += 1
                current_clock_time = self.node.get_clock().now().to_msg()
                latency_ns = current_clock_time.sec * 1e9 + \
                    current_clock_time.nanosec - current_timestamp_ns
                latency_ms = latency_ns / 1e6
                if topic in self._output_topics:
                    self.node.get_logger().info(f'{topic}: {latency_ms} ms')
                if latency_ms > self._max_latency_ms:
                    failure_latency_msg = f'Latency for topic {topic} is {latency_ms} ms higher' \
                                          f'than max allowed latency of {self._max_latency_ms} ms'
                    if_failure_latency = True

                self._data_dict[topic]['latency_ms'].append(latency_ms)

            # If time elapsed is greater than 10 second, calculate hz of each topic
            if time.time() - start_time > self._time_to_run:
                break

        # Plot latency of each topic, save fig to output_dir, different colors for each topic
        plt.figure()
        for topic in self._topics_to_check:
            plt.plot(self._data_dict[topic]['latency_ms'], label=topic)
        plt.legend()
        plt.savefig(os.path.join(self._output_dir, 'latency_plot_robot_segmentor_pol.png'))
        plt.close()

        # First check if each topic has received messages
        for topic in self._topics_to_check:
            if self._data_dict[topic]['num_messages'] == 0:
                self.node.get_logger().error(f'No messages received for topic: {topic}')
                self.fail(f'No messages received for topic: {topic}')

        # The check avg diff between consecutive timestamps is less than 1/hz of the topic
        for topic in self._topics_to_check:

            if topic not in self._output_topics:
                continue

            # Make sure no timestamp arra is of size 0, if yes then fail
            if len(self._data_dict[topic]['timestamp_ns']) == 0:
                self.node.get_logger().error(f'No timestamps received for topic: {topic}')
                self.fail(f'No timestamps received for topic: {topic}')

            avg_diff = np.mean(np.diff(self._data_dict[topic]['timestamp_ns']))
            avg_diff_ms = avg_diff / 1e6
            if avg_diff_ms > 1000:  # 1 second
                self.node.get_logger().error(f'Avg diff between consecutive timestamps for topic'
                                             f'{topic} is {avg_diff_ms} milliseconds')
                self.fail(f'Avg diff between consecutive timestamps for topic'
                          f'{topic} is {avg_diff_ms} milliseconds')

        # Check if latency is higher for output topics
        if if_failure_latency:
            self.node.get_logger().error(failure_latency_msg)
            self.fail(failure_latency_msg)


class IsaacRosManipulationRobotSegmentorPolTest(ManipulatorRobotSegmentorPolTest):
    """Test to verify robot segmentor is running at required hz."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  output_dir: str,
                                  intrinsics_topics: list[str],
                                  input_topics: list[str],
                                  output_topics: list[str],
                                  time_to_run: int,
                                  max_latency_ms: int,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        cls._intrinsics_topics = intrinsics_topics
        cls._input_topics = input_topics
        cls._output_topics = output_topics
        cls._time_to_run = time_to_run
        cls._max_latency_ms = max_latency_ms
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace()
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class DisableNvbloxTest(IsaacROSBaseTest):
    """Verify nvblox and robot segmentor nodes are not running when nvblox is disabled."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_nvblox_nodes_not_running(self):
        """Verify nvblox and robot segmentor nodes are not running when nvblox is disabled."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        time.sleep(self._node_startup_delay)

        running_nodes = self.node.get_node_names()
        self.node.get_logger().info(f'Running nodes: {running_nodes}')

        # Check that nvblox node is not running
        nvblox_node = 'nvblox_node'
        self.assertNotIn(
            nvblox_node,
            running_nodes,
            f'Nvblox node {nvblox_node} is running when it should be disabled'
        )

        # Check that robot segmentor node is not running
        robot_segmentor_node = 'robot_segmenter_1'
        self.assertNotIn(
            robot_segmentor_node,
            running_nodes,
            f'Robot segmentor node {robot_segmentor_node} is running when it should be disabled'
        )

        self.node.get_logger().info(
            'Successfully verified nvblox and robot segmentor nodes are not running'
        )


class DriverTest(IsaacROSBaseTest):
    """Verify driver nodes are running."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _enable_pick_and_place: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  enable_pick_and_place: bool):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._enable_pick_and_place = enable_pick_and_place
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_driver_nodes_running(self):
        """Verify driver nodes are running."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        time.sleep(self._node_startup_delay)

        running_nodes = self.node.get_node_names()
        self.node.get_logger().info(f'Running nodes: {running_nodes}')

        if self._use_sim_time:
            control_nodes_that_should_be_running = [
                'move_group',
                'moveit',
                'moveit_simple_controller_manager',
                'controller_manager',
                'joint_state_broadcaster',
                'joint_trajectory_controller',
                'robot_state_publisher',
                'joint_parser',
                'isaac_sim_gripper_action_server',
            ]
        else:
            control_nodes_that_should_be_running = [
                'move_group',
                'moveit',
                'moveit_simple_controller_manager',
                'controller_manager',
                'joint_state_broadcaster',
                'io_and_status_controller',
                'speed_scaling_state_broadcaster',
                'force_torque_sensor_broadcaster',
                'tcp_pose_broadcaster',
                'ur_configuration_controller',
                'joint_trajectory_controller',
                'controller_stopper',
                'robot_state_publisher',
            ]

            if self._enable_pick_and_place:
                control_nodes_that_should_be_running.append('robotiq_gripper_controller')
                control_nodes_that_should_be_running.append('robotiq_activation_controller')

        for node in control_nodes_that_should_be_running:
            self.assertIn(
                node,
                running_nodes,
                f'{node} is not running when it should be running'
            )


class FlexivDriverTest(IsaacROSBaseTest):
    """
    Verify Flexiv Rizon driver nodes are running.

    Flexiv mirror of :class:`DriverTest`. Asserts the control-plane and MoveIt
    nodes that the Flexiv Rizon real-robot driver stack brings up when routed
    through the bringup plugin (``drivers.launch.py`` -> Flexiv real driver).
    """

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_driver_nodes_running(self):
        """Verify Flexiv driver nodes are running."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        time.sleep(self._node_startup_delay)

        running_nodes = self.node.get_node_names()
        self.node.get_logger().info(f'Running nodes: {running_nodes}')

        # Core nodes brought up by flexiv_rizon_real_driver.launch.py on every run,
        # independent of use_fake_hardware/load_gripper gating.
        control_nodes_that_should_be_running = [
            'move_group',
            'controller_manager',
            'robot_state_publisher',
            'joint_state_publisher',
            'joint_state_broadcaster',
            'rizon_arm_controller',
        ]

        for node in control_nodes_that_should_be_running:
            self.assertIn(
                node,
                running_nodes,
                f'{node} is not running when it should be running'
            )


class IsaacRosManipulationServersPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_cycles: int = 10
    _use_sim_time: bool = False
    _mesh_file_path: str = ''
    _run_test: bool = False
    _detection_client: ActionClient = None
    _pose_client: ActionClient = None
    _segmentation_client: ActionClient = None
    _add_mesh_client: ActionClient = None
    _clear_objects_client: ActionClient = None
    _initial_hint: Dict = None

    # Variables to track object detection backend, segmentaiton backend and pose estimation backend
    _is_segment_anything_object_detection_enabled: bool = False
    _is_segment_anything_segmentation_enabled: bool = False
    _is_rt_detr_object_detection_enabled: bool = False

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self.failure_count = 0
        self.total_count = 0

        self._detection_client = ActionClient(self.node, GetObjects, '/get_objects')
        self._get_objects_goal_handle = None
        self._get_objects_result_received = False
        self._get_objects_result_future = None
        self._get_objects_detected_object_id = None

        self._get_object_pose_client = ActionClient(self.node, GetObjectPose, '/get_object_pose')
        self._get_object_pose_goal_handle = None
        self._get_object_pose_result_received = False
        self._get_object_pose_result_future = None

        self._segmentation_client = ActionClient(self.node, AddSegmentationMask,
                                                 '/add_segmentation_mask')
        self._segmentation_goal_handle = None
        self._segmentation_result_received = False
        self._segmentation_result_future = None

        self._add_mesh_client = self.node.create_client(AddMeshToObject, '/add_mesh_to_object')
        self._add_mesh_goal_handle = None
        self._add_mesh_result_received = False
        self._add_mesh_result_future = None

        self._clear_objects_client = self.node.create_client(ClearObjects, '/clear_objects')
        self._clear_objects_goal_handle = None
        self._clear_objects_result_received = False
        self._clear_objects_result_future = None

        self._initial_hint_point_msg = Point()
        if self._initial_hint is not None:
            self._initial_hint_point_msg.x = self._initial_hint['x']
            self._initial_hint_point_msg.y = self._initial_hint['y']
            self._initial_hint_point_msg.z = self._initial_hint['z']

    def trigger_get_objects_goal(self, initial_hint: None | Point = None):
        """Send a goal to the object detection action server."""
        self.node.get_logger().info('Waiting for action server for object detection...')
        if not self._detection_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error(
                'Action server for object detection not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        if initial_hint is not None:
            goal_msg.initial_hint = initial_hint
            goal_msg.use_initial_hint = True

        self.node.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._detection_client.send_goal_async(
            goal_msg, feedback_callback=self.get_objects_feedback_callback)
        send_goal_future.add_done_callback(self.get_objects_goal_response_callback)

    def get_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_objects_goal_handle = future.result()
        if not self._get_objects_goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.node.get_logger().info('Goal accepted by the action server.')
        self._get_objects_result_future = self._get_objects_goal_handle.get_result_async()
        self._get_objects_result_future.add_done_callback(self.get_objects_result_callback)

    def get_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        object_ids = []
        if result.objects:
            self.node.get_logger().info('Objects detected, the objects id are below:')
            for object_obj in result.objects:
                self.node.get_logger().info(f'Object: {object_obj.object_id}')
                object_ids.append(object_obj.object_id)

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self._get_objects_detected_object_id = result.objects[0].object_id
            self._get_objects_result_received = True
        else:
            self.node.get_logger().error(f'Action failed with status: {status}')
            self._get_objects_result_received = False

    def trigger_get_object_pose_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.node.get_logger().info('Waiting for pose estimation action server...')
        if not self._get_object_pose_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Pose estimation action server not available')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._get_object_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.get_object_pose_feedback_callback)
        send_goal_future.add_done_callback(self.get_object_pose_goal_response_callback)

    def get_object_pose_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_object_pose_goal_handle = future.result()
        if not self._get_object_pose_goal_handle.accepted:
            self.node.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Pose estimation goal accepted by the action server.')
        self._get_object_pose_result_future = self._get_object_pose_goal_handle.get_result_async()
        self._get_object_pose_result_future.add_done_callback(self.get_object_pose_result_callback)

    def get_object_pose_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_object_pose_result_callback(self, future):
        """Handle the result from the action server."""
        status = future.result().status

        # Log result to INFO
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Object pose received for object id')
            self._get_object_pose_result_received = True
        else:
            self.node.get_logger().error(f'Pose estimation action failed with status: {status}')
            self._get_object_pose_result_received = False

    def trigger_get_segmented_object_goal(self, object_id):
        """Send a goal to the object segmentation action server."""
        self.node.get_logger().info('Waiting for segmentation action server...')
        if not self._segmentation_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Segmentation action server not available after waiting')
            return

        goal_msg = AddSegmentationMask.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to get segmented object for ID: {object_id}')
        send_goal_future = self._segmentation_client.send_goal_async(
            goal_msg, feedback_callback=self.get_segmented_object_feedback_callback)
        send_goal_future.add_done_callback(self.get_segmented_object_goal_response_callback)

    def get_segmented_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._segmentation_goal_handle = future.result()
        if not self._segmentation_goal_handle.accepted:
            self.node.get_logger().error('Segmentation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Segmentation goal accepted by the action server.')
        self._segmentation_result_future = self._segmentation_goal_handle.get_result_async()
        self._segmentation_result_future.add_done_callback(
            self.get_segmented_object_result_callback)

    def get_segmented_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Segmentation Feedback: {feedback_msg.feedback}')

    def get_segmented_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Segmented object received: {result}')
            self._segmentation_result_received = True
        else:
            self.node.get_logger().error(f'Segmentation action failed with status: {status}')
            self._segmentation_result_received = False

    def trigger_add_mesh_to_object_request(self, object_id, mesh_file_path):
        request = AddMeshToObject.Request()
        request.object_ids = [object_id]
        request.mesh_file_paths = [mesh_file_path]
        future = self._add_mesh_client.call_async(request)
        return future

    def send_clear_objects_request(self, object_ids=None):
        request = ClearObjects.Request()
        if object_ids:
            request.object_ids = object_ids
        future = self._clear_objects_client.call_async(request)
        return future

    def add_mesh_to_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._add_mesh_goal_handle = future.result()
        if not self._add_mesh_goal_handle.accepted:
            self.node.get_logger().error('Add mesh goal was rejected by the action server.')
            return

        self.node.get_logger().info('Add mesh goal accepted by the action server.')
        self._add_mesh_result_future = self._add_mesh_goal_handle.get_result_async()
        self._add_mesh_result_future.add_done_callback(self.add_mesh_to_object_result_callback)

    def add_mesh_to_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Add Mesh Feedback: {feedback_msg.feedback}')

    def add_mesh_to_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        self.node.get_logger().info(f'Add mesh result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Mesh added to object: {result.object_id}')
            self._add_mesh_result_received = True
        else:
            self.node.get_logger().error(f'Add mesh action failed with status: {status}')
            self._add_mesh_result_received = False

    def clear_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._clear_objects_goal_handle = future.result()
        if not self._clear_objects_goal_handle.accepted:
            self.node.get_logger().error('Clear objects goal was rejected by the action server.')
            return

        self.node.get_logger().info('Clear objects goal accepted by the action server.')
        self._clear_objects_result_future = self._clear_objects_goal_handle.get_result_async()
        self._clear_objects_result_future.add_done_callback(self.clear_objects_result_callback)

    def clear_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Clear Objects Feedback: {feedback_msg.feedback}')

    def clear_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        self.node.get_logger().info(f'Clear objects result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Objects cleared successfully')
            self._clear_objects_result_received = True
        else:
            self.node.get_logger().error(f'Clear objects action failed with status: {status}')
            self._clear_objects_result_received = False

    def wait_for_result(self, result_var_name: str, timeout: float = 10.0):
        start_time = self.node.get_clock().now()
        while rclpy.ok():
            # Check the actual instance variable dynamically
            if result_var_name and getattr(self, result_var_name):
                return True

            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                # self.fail(f'Timeout waiting for {result_var_name}')
                self.failure_count += 1
                return False
            self.total_count += 1

            rclpy.spin_once(self.node, timeout_sec=0.1)

    def wait_for_service_result(self, future, timeout=10.0):
        start_time = self.node.get_clock().now()
        while not future.done():
            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.node.get_logger().error('Timeout waiting for service result')
                return None
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return future.result()

    def test_for_servers_pol(self):
        """Test that verifies that servers action calls under multiple cycles."""
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.setUpClients()

        self.node.get_logger().info('Starting test for manipulator servers POL')

        time.sleep(10)  # Give servers some additional time to come up

        for _ in range(self._num_cycles):

            initial_hint = None
            if self._is_segment_anything_object_detection_enabled:
                initial_hint = self._initial_hint_point_msg

            # First detect object
            self.trigger_get_objects_goal(initial_hint)

            # Wait for the detection result with a timeout
            self.wait_for_result('_get_objects_result_received',
                                 timeout=self._max_timeout_time_for_action_call)
            if not self._get_objects_result_received:
                self.node.get_logger().error('Object detection action failed')
                self.failure_count += 1
                continue

            self.total_count += 1
            # If segment anything is enabled, we need to get the segmented object first
            if self._is_segment_anything_segmentation_enabled:
                self.trigger_get_segmented_object_goal(self._get_objects_detected_object_id)
                self.wait_for_result('_segmentation_result_received',
                                     timeout=self._max_timeout_time_for_action_call)

            # Now add mesh to object
            if self._is_segment_anything_segmentation_enabled:
                future = self.trigger_add_mesh_to_object_request(
                    self._get_objects_detected_object_id, self._mesh_file_path)
                result = self.wait_for_service_result(
                    future, timeout=self._max_timeout_time_for_action_call)
                self.node.get_logger().info(f'Add mesh to object result: {result}')
                if result is None:
                    self.node.get_logger().error('Add mesh to object action failed')
                    self.failure_count += 1
                    continue
                self.total_count += 1

            # Now get object pose
            self.trigger_get_object_pose_goal(self._get_objects_detected_object_id)
            self.wait_for_result('_get_object_pose_result_received',
                                 timeout=self._max_timeout_time_for_action_call)

            # Now clear objects
            future = self.send_clear_objects_request()
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.node.get_logger().info(f'Clear objects result: {result}')
            if result is None:
                self.node.get_logger().error('Clear objects action failed')
                self.failure_count += 1
                continue
            self.total_count += 1

            # Reset for next iteration
            self._get_objects_result_received = False
            self._segmentation_result_received = False
            self._add_mesh_result_received = False
            self._get_object_pose_result_received = False
            self._clear_objects_result_received = False

        self.node.get_logger().info(f'Failure count: {self.failure_count}')
        assert self.failure_count / self.total_count < 0.3, f'Pose estimation action failed' \
            f'{self.failure_count}/{self.total_count} times'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  num_cycles: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  is_segment_anything_object_detection_enabled: bool,
                                  is_segment_anything_segmentation_enabled: bool,
                                  is_rt_detr_object_detection_enabled: bool,
                                  initial_hint: Dict,
                                  mesh_file_path: str,
                                  output_dir: str,
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._num_cycles = num_cycles
        cls._use_sim_time = use_sim_time
        cls._is_segment_anything_object_detection_enabled = \
            is_segment_anything_object_detection_enabled
        cls._is_segment_anything_segmentation_enabled = is_segment_anything_segmentation_enabled
        cls._is_rt_detr_object_detection_enabled = is_rt_detr_object_detection_enabled
        cls._initial_hint = initial_hint
        cls._output_dir = output_dir
        cls._mesh_file_path = mesh_file_path

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class CollectTargetLessCalibrationDataTest(IsaacROSBaseTest):
    """Test for Collect Target Less Calibration Data."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0
    _camera_topic_names: List[str] = []
    _camera_info_topic_names: List[str] = []
    _base_frame_name: str = ''
    _joints_to_query: List[str] = []
    _urdf_file_path: str = ''
    _output_dir: str = ''
    _joint_topic_name: str = '/joint_states'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  nodes: list[Node],
                                  use_sim_time: bool,
                                  node_startup_delay: float,
                                  camera_topic_names: List[str],
                                  camera_info_topic_names: List[str],
                                  base_frame_name: str,
                                  joints_to_query: List[str],
                                  urdf_file_path: str,
                                  output_dir: str,
                                  joint_topic_name: str):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._camera_topic_names = camera_topic_names
        cls._camera_info_topic_names = camera_info_topic_names
        cls._base_frame_name = base_frame_name
        cls._joints_to_query = joints_to_query
        cls._urdf_file_path = urdf_file_path
        cls._output_dir = output_dir
        cls._joint_topic_name = joint_topic_name

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUpClients(self) -> None:
        """Set up before each test method."""
        # Setup subscibers to images and camera info topics
        self.received_messages = {}
        topic_name_tuples = []

        self._received_image_topics = {}
        self._received_camera_info_topics = {}
        self._received_joint_transforms = {}

        for camera_topic_name in self._camera_topic_names:
            topic_name_tuples.append((camera_topic_name, Image))
            self._received_image_topics[camera_topic_name] = False

        for camera_info_topic_name in self._camera_info_topic_names:
            topic_name_tuples.append((camera_info_topic_name, CameraInfo))
            self._received_camera_info_topics[camera_info_topic_name] = False

        topic_name_tuples.append((self._joint_topic_name, JointState))

        for joint_name in self._joints_to_query:
            self._received_joint_transforms[joint_name] = False

        self._subs = self.create_logging_subscribers(
            topic_name_tuples,
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        # Set up TF subsriber
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        self._bridge = CvBridge()

    def _save_image_to_output_dir(self, data: Image, output_dir: str, camera_topic_name: str):
        """Save image to output directory."""
        # Save image to output directory
        # Use cvbridge to convert the image to cv2 format
        os.makedirs(output_dir, exist_ok=True)
        cv_image = np.array(self._bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        image_path = os.path.join(output_dir,
                                  camera_topic_name.replace('/', '_') + '.png')
        cv2.imwrite(image_path, cv_image)
        self.node.get_logger().info(f'Saved image to {image_path}')

    def _save_camera_info_to_output_dir(self,
                                        data: CameraInfo,
                                        output_dir: str,
                                        camera_info_topic_name: str):
        """Save camera info to output directory."""

        def camera_info_to_dict(camera_info_msg: CameraInfo) -> Dict[str, Any]:
            """Convert ROS CameraInfo message to dictionary for JSON serialization."""
            return {
                'header': {
                    'frame_id': camera_info_msg.header.frame_id
                },
                'width': camera_info_msg.width,
                'height': camera_info_msg.height,
                'distortion_model': camera_info_msg.distortion_model,
                'D': list(camera_info_msg.d),
                'K': list(camera_info_msg.k),
                'R': list(camera_info_msg.r),
                'P': list(camera_info_msg.p)
            }
        # Save camera info to output directory
        os.makedirs(output_dir, exist_ok=True)
        camera_info_path = os.path.join(output_dir,
                                        camera_info_topic_name.replace('/', '_') + '.json')
        with open(camera_info_path, 'w') as f:
            json.dump(camera_info_to_dict(data), f)
        self.node.get_logger().info(f'Saved camera info to {camera_info_path}')

    def _save_joint_transform_to_output_dir(self,
                                            data: TransformStamped,
                                            output_dir: str,
                                            joint_name: str):
        """Save joint transform to output directory."""
        # Save joint transform to output directory
        os.makedirs(output_dir, exist_ok=True)
        joint_transform_path = os.path.join(output_dir, joint_name + '.json')

        # Convert the transform to a dictionary
        transform_dict = {
            'translation': {
                'x': data.transform.translation.x,
                'y': data.transform.translation.y,
                'z': data.transform.translation.z
            },
            'rotation': {
                'x': data.transform.rotation.x,
                'y': data.transform.rotation.y,
                'z': data.transform.rotation.z,
                'w': data.transform.rotation.w
            },
            'parent_frame': data.header.frame_id,
            'child_frame': data.child_frame_id
        }

        with open(joint_transform_path, 'w') as f:
            json.dump(transform_dict, f)
        self.node.get_logger().info(f'Saved joint transform to {joint_transform_path}')

    def test_run_calibration_data_collection_test(self):

        self.setUpClients()

        # spin the node for 1000 seconds
        joint_angles_saved = False
        start_time = time.time()
        max_timeout_time_for_action_call = 10  # 10 seconds to get the data
        while True:
            if time.time() - start_time > max_timeout_time_for_action_call:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(1)
            self.node.get_logger().info('Spinning the node for 1000 seconds')

            # Now first check for all camera topics to have received messages
            for camera_topic_name in self._camera_topic_names:
                if not self._received_image_topics[camera_topic_name]:
                    # Check if the camera info topic has received a message
                    if (
                        self.received_messages[camera_topic_name] is not None
                        and len(self.received_messages[camera_topic_name]) > 0
                    ):
                        self._received_image_topics[camera_topic_name] = True
                        # Save image in output directory by first checking it to cv2.
                        self._save_image_to_output_dir(
                            data=self.received_messages[camera_topic_name][-1],
                            output_dir=self._output_dir,
                            camera_topic_name=camera_topic_name
                        )
                else:
                    continue

            # Now check for all camera info topics to have received messages
            for camera_info_topic_name in self._camera_info_topic_names:
                if not self._received_camera_info_topics[camera_info_topic_name]:
                    if (
                        self.received_messages[camera_info_topic_name] is not None
                        and len(self.received_messages[camera_info_topic_name]) > 0
                    ):
                        self._received_camera_info_topics[camera_info_topic_name] = True
                        # Save camera info in output directory by first checking it to cv2.
                        self._save_camera_info_to_output_dir(
                            data=self.received_messages[camera_info_topic_name][-1],
                            output_dir=self._output_dir,
                            camera_info_topic_name=camera_info_topic_name
                        )
                else:
                    continue

            if not joint_angles_saved:
                if (
                    self.received_messages[self._joint_topic_name] is not None
                    and len(self.received_messages[self._joint_topic_name]) > 0
                ):
                    joint_state = self.received_messages[self._joint_topic_name][0]
                    json_dict = {
                        'positions': list(joint_state.position),
                        'names': joint_state.name
                    }
                    with open(f'{self._output_dir}/joint_angles.json', 'w') as f:
                        json.dump(json_dict, f)
                    joint_angles_saved = True

            # Now check all joints and check if their transform is available w.r.t base link frame
            for joint_name in self._joints_to_query:
                if self._received_joint_transforms[joint_name]:
                    continue
                # Get the transform from the TF buffer
                try:
                    transform = self._tf_buffer.lookup_transform(
                        self._base_frame_name,
                        joint_name,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                except Exception as e:
                    self.node.get_logger().warning(
                        f'Failed to get transform for {joint_name}: {e}')
                    continue

                if transform is not None:
                    # Save joint transform in output directory by first checking it to cv2.
                    self._save_joint_transform_to_output_dir(
                        data=transform,
                        output_dir=self._output_dir,
                        joint_name=joint_name
                    )
                    self._received_joint_transforms[joint_name] = True


class PoseEstimationDifferentDepthBackendsTest(IsaacROSBaseTest):
    """Test for Isaac ROS Test Hardcoded Poses."""

    DEFAULT_NAMESPACE = ''
    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0
    _poses: list[dict] = []
    _hang_test_after_completion: bool = False

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Pose): The grasp pose to publish.
            parent_frame (str): The parent frame.
            child_frame (str): The child frame.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.node.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.node.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            raise e

    def setup_clients(self):

        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)

    def test_for_different_depth_backends(self):
        if not self._run_test:
            self.node.get_logger().warning('RUN_TEST is not set to true')
            return

        self.setup_clients()
        self.node.get_logger().info('Starting test for different depth backends')

        for idx in range(len(self._poses)):

            pose_obj = self._poses[idx]
            pose = Pose()
            pose.position.x = pose_obj['position']['x']
            pose.position.y = pose_obj['position']['y']
            pose.position.z = pose_obj['position']['z']
            pose.orientation.x = pose_obj['orientation']['x']
            pose.orientation.y = pose_obj['orientation']['y']
            pose.orientation.z = pose_obj['orientation']['z']
            pose.orientation.w = pose_obj['orientation']['w']

            self.node.get_logger().info(f'Pose: {pose_obj}')

            # First publish on tf static under a name.
            self.publish_pose_on_tf_static(
                pose,
                parent_frame=pose_obj['frame_id'],
                child_frame=pose_obj['pose_name_for_tf']
            )

            # spin a few times.
            for _ in range(100):
                rclpy.spin_once(self.node, timeout_sec=0.1)

        if self._hang_test_after_completion:
            time.sleep(1000)

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  poses: list[dict],
                                  hang_test_after_completion: bool):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._poses = poses
        cls._hang_test_after_completion = hang_test_after_completion
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

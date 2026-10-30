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

import time
from typing import Any, Dict, Optional

from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class BaseServiceBehavior(py_trees.behaviour.Behaviour):
    """
    Base class for behaviors that use ROS2 services.

    Implements common functionality for service clients and state management.

    Parameters
    ----------
    name : str
        Name of the behavior
    service_type : type
        The ROS2 service type to use
    service_name : str
        Name of the service to connect to
    blackboard_keys : dict, optional
        Dictionary of blackboard keys to register with their access permissions

    Note
    ----
    Server timeout configuration is read from the blackboard 'server_timeout_config' key.
    This dictionary should contain:
    - startup_server_timeout_sec: Max time to wait for server during startup (None = wait forever)
    - runtime_retry_timeout_sec: Max time to retry server connection during execution
    - server_check_interval_sec: Interval between server availability checks in seconds
      (also determines how often progress is logged)

    """

    def __init__(
        self,
        name: str,
        service_type: Any,
        service_name: str,
        blackboard_keys: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()

        # Register server timeout config access
        self.blackboard.register_key(
            key='server_timeout_config',
            access=py_trees.common.Access.READ
        )

        # Register blackboard keys
        if blackboard_keys:
            for key, access in blackboard_keys.items():
                self.blackboard.register_key(key=key, access=access)

        # ROS2 Node and Service Client variables
        self.node = None
        self._service_client = None
        self._service_type = service_type
        self._service_name = service_name

        # Service state tracking
        self._service_state = BehaviorStatus.IDLE
        self._last_server_check_time = 0.0

        # Service response
        self._service_response = None

    def setup(self, **kwargs):
        """
        Set up the behavior by creating the service client and waiting for server availability.

        This is called once when the behavior tree is constructed.
        Waits for the service server to become available with a long timeout or indefinitely.

        Raises
        ------
            KeyError: If 'node' is not in kwargs
            RuntimeError: If service server is not available within startup timeout

        """
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e

        # Read timeout configuration from blackboard
        if (self.blackboard.exists('server_timeout_config') and
                self.blackboard.server_timeout_config):
            server_timeout_config = self.blackboard.server_timeout_config
            self._startup_server_timeout_sec = server_timeout_config.get(
                'startup_server_timeout_sec')
            self._runtime_retry_timeout_sec = server_timeout_config.get(
                'runtime_retry_timeout_sec', 10.0)
            self._server_check_interval_sec = server_timeout_config.get(
                'server_check_interval_sec', 20.0)
        else:
            # Raise error if server timeout config is not available
            raise RuntimeError(
                f"server_timeout_config not available on blackboard for behavior '{self.name}'. "
                f'Ensure blackboard is properly initialized before tree setup.')

        callback_group = MutuallyExclusiveCallbackGroup()
        self._service_client = self.node.create_client(
            self._service_type,
            self._service_name,
            callback_group=callback_group
        )

        # Wait for server during setup with long timeout or indefinitely
        self._wait_for_server_during_setup()

    def initialise(self):
        """
        Reset the state variables for this run.

        This is called every time the behavior starts to run.
        """
        # Reset service state
        self._service_state = BehaviorStatus.IDLE
        self._service_response = None

        # State variables for server retry logic
        self._retry_start_time = None
        self._last_server_check_time = 0.0

    def update(self):
        """
        Perform the base implementation of the behavior update.

        Implements retry logic for temporary server unavailability.
        Derived classes should override this with their specific logic,
        but can call this method for common state handling.

        Returns
        -------
        py_trees.common.Status
            FAILURE if the server is unavailable after retry timeout,
            RUNNING if the service call is in progress or retrying server connection,
            SUCCESS is never returned (derived classes handle success)

        """
        # Call retry logic if we're in retry mode OR server is down
        # This ensures _reset_retry_tracking() gets called when server recovers between ticks
        if self._retry_start_time is not None or not self._service_client.service_is_ready():
            retry_status = self._handle_server_retry()
            if retry_status == py_trees.common.Status.RUNNING:
                return py_trees.common.Status.RUNNING
            elif retry_status == py_trees.common.Status.FAILURE:
                return py_trees.common.Status.FAILURE
            # If retry_status is SUCCESS, server is recovered, continue with normal flow

        # Handle service failure case
        if self._service_state == BehaviorStatus.FAILED:
            self.node.get_logger().error(
                f'[{self.name}] Service call to {self._service_name} failed')
            return py_trees.common.Status.FAILURE

        # Remaining states (IDLE, IN_PROGRESS, SUCCEEDED) are handled by derived classes
        return py_trees.common.Status.RUNNING

    def _wait_for_server_during_setup(self) -> None:
        """
        Wait for service server during setup phase.

        Blocks until server is available or startup timeout exceeded.
        Logs progress every server_check_interval_sec seconds.

        Raises
        ------
            RuntimeError: If startup timeout is exceeded

        """
        self.node.get_logger().info(f'Waiting for {self._service_name} service server...')
        start_time = time.time()

        while True:
            server_available = self._service_client.wait_for_service(
                timeout_sec=self._server_check_interval_sec)

            if server_available:
                self.node.get_logger().info(f'{self._service_name} service is available')
                return

            elapsed_time = time.time() - start_time
            if self._check_timeout_exceeded(elapsed_time, self._startup_server_timeout_sec):
                raise RuntimeError(
                    f'{self._service_name} service not available after '
                    f'{self._startup_server_timeout_sec}s during startup')

            timeout_msg = ('indefinitely' if self._startup_server_timeout_sec is None
                           else f'for up to {self._startup_server_timeout_sec}s')
            self.node.get_logger().warning(
                f'Still waiting for {self._service_name} service '
                f'({elapsed_time:.0f}s elapsed, waiting {timeout_msg})...')

    def _handle_server_retry(self) -> py_trees.common.Status:
        """
        Handle service unavailability during execution with retry logic.

        Manages retry process when service server becomes unavailable:
        1. Initializes retry tracking on first call
        2. Checks server availability at regular intervals
        3. Returns SUCCESS if server is recovered
        4. Returns RUNNING while retrying (timeout not reached)
        5. Returns FAILURE if retry timeout is exceeded

        Returns
        -------
        py_trees.common.Status
            SUCCESS if server is recovered, RUNNING if retrying, FAILURE if timeout exceeded

        """
        current_time = time.time()

        # Initialize retry tracking on first call when server becomes unavailable
        if self._retry_start_time is None:
            self._retry_start_time = current_time
            self.node.get_logger().warning(
                f'{self._service_name} service unavailable, starting retry phase ...')

        # Check if server has become available again
        if self._service_client.service_is_ready():
            self._reset_retry_tracking()
            return py_trees.common.Status.SUCCESS

        # Log retry progress at regular intervals to avoid log spam
        time_since_last_check = current_time - self._last_server_check_time
        if time_since_last_check >= self._server_check_interval_sec:
            self._last_server_check_time = current_time
            self.node.get_logger().warning(
                f'{self._service_name} service unavailable, retrying...')

        # Check if we've exceeded the retry timeout
        retry_elapsed = current_time - self._retry_start_time
        if self._check_timeout_exceeded(retry_elapsed, self._runtime_retry_timeout_sec):
            self.node.get_logger().error(
                f'{self._service_name} service unavailable for {retry_elapsed:.1f}s, '
                f'exceeded retry timeout of {self._runtime_retry_timeout_sec}s')
            return py_trees.common.Status.FAILURE

        # Server still unavailable but timeout not reached - continue retrying
        return py_trees.common.Status.RUNNING

    def _reset_retry_tracking(self) -> None:
        """Reset the retry tracking variables and handle state recovery."""
        self._retry_start_time = None
        self.node.get_logger().info(f'{self._service_name} service recovered')

        # Handle interrupted service calls with derived class involvement
        if self._service_state == BehaviorStatus.IN_PROGRESS:
            self.node.get_logger().warning(
                f'{self._service_name} service call was interrupted by server unavailability')

            # Let derived classes handle their specific recovery needs
            target_state = self.handle_server_recovery()

            if target_state == BehaviorStatus.IDLE:
                self.node.get_logger().info(
                    f'Resetting service state to IDLE - will attempt to make a new call to '
                    f'{self._service_name}')
                self._service_state = BehaviorStatus.IDLE
                self._service_response = None
            elif target_state == BehaviorStatus.FAILED:
                self.node.get_logger().error(
                    f'Marking service as FAILED due to server interruption - behavior will fail '
                    f'and not attempt to make a new call to {self._service_name}')
                self._service_state = BehaviorStatus.FAILED
                self._service_response = None

    def _check_timeout_exceeded(self, elapsed_time: float, timeout: Optional[float]) -> bool:
        """
        Check if the elapsed time has exceeded the specified timeout.

        Parameters
        ----------
        elapsed_time : float
            Time elapsed in seconds
        timeout : Optional[float]
            Timeout value in seconds (None means no timeout)

        Returns
        -------
        bool
            True if timeout is set and exceeded, False otherwise

        """
        # If timeout is None, there's no timeout (wait forever)
        if timeout is None:
            return False
        # Otherwise, check if elapsed time exceeds the timeout
        return elapsed_time >= timeout

    def handle_server_recovery(self) -> BehaviorStatus:
        """
        Handle server recovery for interrupted service calls.

        This method is called when a service server becomes available again after
        an interruption while a service call was IN_PROGRESS. Child classes can override
        this method to implement custom recovery logic based on their specific needs.

        Returns
        -------
        BehaviorStatus
            IDLE - Reset to IDLE state, allowing the behavior to attempt making a new service call.
                   Use this for stateless services where retrying the call is safe.
            FAILED - Mark the behavior as FAILED, which will cause the behavior tree to fail.
                     Use this for services with side effects where the interruption makes
                     recovery unsafe.

        """
        # Default: Reset to IDLE to allow retry - services are typically more stateless
        # than actions
        return BehaviorStatus.IDLE

    # Public methods for child classes to use
    def call_service(self, request):
        """
        Make a service call and process the result.

        Returns
        -------
        bool
            True if the call was successfully initiated, False otherwise

        """
        if not self._service_client.service_is_ready():
            self.node.get_logger().error(
                f'[{self.name}] {self._service_name} service not available')
            self._service_state = BehaviorStatus.FAILED
            return False

        try:
            # Make the service call
            future = self._service_client.call_async(request)
            future.add_done_callback(self._service_done_callback)
            self._service_state = BehaviorStatus.IN_PROGRESS
            return True

        except Exception as e:
            self.node.get_logger().error(
                f'[{self.name}] Error calling service {self._service_name}: {str(e)}')
            self._service_state = BehaviorStatus.FAILED
            return False

    # Protected method for child classes to set failure state
    def set_service_failed(self):
        """Set the service state to FAILED. Used by child classes when they encounter errors."""
        self._service_state = BehaviorStatus.FAILED

    # Getter methods
    def get_service_state(self):
        """Get the current service state."""
        return self._service_state

    def get_service_response(self):
        """Get the service response."""
        return self._service_response

    # Private methods
    def _service_done_callback(self, future):
        """Process the result of a service call."""
        try:
            self._service_response = future.result()
            self._service_state = BehaviorStatus.SUCCEEDED
        except Exception as e:
            self.node.get_logger().error(
                f'[{self.name}] Service call to {self._service_name} failed: {str(e)}')
            self._service_state = BehaviorStatus.FAILED
            self._service_response = None

    def terminate(self, new_status):
        """Clean up when this behavior finishes running."""
        # Log termination
        self.node.get_logger().debug(
            f'{self.name} behavior terminated with status {new_status}')

        # Clean up
        self._service_state = BehaviorStatus.IDLE
        self._service_response = None

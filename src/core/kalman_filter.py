"""
Kalman Filter Module

This module handles Kalman filtering for object tracking and prediction.
"""

import cv2
import numpy as np


class KalmanFilterManager:
    """
    Manages Kalman filters for multiple objects.
    """

    def __init__(self):
        """Initialize the Kalman filter manager."""
        # Kalman filter for each object
        self.kalman_filters = {}  # Dictionary to store Kalman filter for each object ID
        self.kalman_initialized = {}  # Track which filters have been initialized
        self.kalman_centers = {}  # Track Kalman-filtered positions
        self.use_kalman = True  # Flag to enable/disable Kalman filtering

    def initialize_kalman_filter(self):
        """
        Initialize a new Kalman filter for tracking.

        Returns:
        --------
        cv2.KalmanFilter
            Initialized Kalman filter
        """
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)

        # Increased process noise for better handling of occlusions
        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32) * 0.1  # Increased from 0.03

        # Decreased measurement noise to trust measurements more when available
        kalman.measurementNoiseCov = np.array([[1, 0],
                                             [0, 1]], np.float32) * 0.005  # Decreased from 0.01

        # Initialize error covariance
        kalman.errorCovPost = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.1

        return kalman

    def update_kalman_filters(self, tracked_objects):
        """
        Update Kalman filters for all tracked objects after regular tracking.

        Parameters:
        -----------
        tracked_objects : dict
            Dictionary of tracked objects
        """
        if not self.use_kalman:
            return

        # Process each tracked object
        for object_id, obj in tracked_objects.items():
            center_x, center_y = obj['center']
            # Create and initialize filter if it doesn't exist
            if object_id not in self.kalman_filters:
                self.kalman_filters[object_id] = self.initialize_kalman_filter()
                # Initialize state with first measurement
                self.kalman_filters[object_id].statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
                self.kalman_initialized[object_id] = True
                self.kalman_centers[object_id] = (center_x, center_y)
                continue

            # For existing filters, predict and then correct with new measurement
            kalman = self.kalman_filters[object_id]

            # First predict
            kalman.predict()

            # Then correct with measurement
            measurement = np.array([[center_x], [center_y]], np.float32)
            kalman.correct(measurement)

            # Store the filtered position
            filtered_x = int(kalman.statePost[0][0])
            filtered_y = int(kalman.statePost[1][0])
            self.kalman_centers[object_id] = (filtered_x, filtered_y)

        # Clean up filters for objects that no longer exist
        for object_id in list(self.kalman_filters.keys()):
            if object_id not in tracked_objects:
                del self.kalman_filters[object_id]
                if object_id in self.kalman_initialized:
                    del self.kalman_initialized[object_id]
                if object_id in self.kalman_centers:
                    del self.kalman_centers[object_id]

    def get_kalman_center(self, object_id):
        """
        Get Kalman-filtered center position for an object, if available.

        Parameters:
        -----------
        object_id : int
            Object ID

        Returns:
        --------
        tuple or None
            (x, y) coordinates if available, None otherwise
        """
        if not self.use_kalman or object_id not in self.kalman_centers:
            return None
        return self.kalman_centers[object_id]

    def get_kalman_velocity(self, object_id):
        """
        Get velocity from Kalman filter for an object, if available.

        Parameters:
        -----------
        object_id : int
            Object ID

        Returns:
        --------
        tuple or None
            (magnitude, (vx, vy)) if available, None otherwise
        """
        if not self.use_kalman or object_id not in self.kalman_filters:
            return None

        kalman = self.kalman_filters[object_id]
        vx = kalman.statePost[2][0]
        vy = kalman.statePost[3][0]
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        return velocity_magnitude, (vx, vy)

    def get_kalman_prediction(self, object_id):
        """
        Get predicted position from Kalman filter for an object.

        Parameters:
        -----------
        object_id : int
            Object ID

        Returns:
        --------
        tuple or None
            (x, y) predicted coordinates if available, None otherwise
        """
        if not self.use_kalman or object_id not in self.kalman_filters:
            return None

        kalman = self.kalman_filters[object_id]
        # Get the predicted state
        prediction = kalman.predict().copy()  # Make a copy to not affect the actual filter
        # Reset the filter state since we don't want this prediction to affect the actual tracking
        kalman.statePost = kalman.statePre.copy()

        # Extract predicted x, y coordinates
        pred_x = int(prediction[0][0])
        pred_y = int(prediction[1][0])

        return (pred_x, pred_y)
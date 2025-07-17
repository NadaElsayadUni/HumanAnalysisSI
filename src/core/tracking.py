"""
Object Tracking Module

This module handles object tracking, ID assignment, and trajectory management.
"""

import cv2
import numpy as np
import random
from collections import defaultdict


class ObjectTracker:
    """
    Handles object tracking, ID assignment, and trajectory management.
    """

    def __init__(self, frame_width, frame_height):
        """
        Initialize the object tracker.

        Parameters:
        -----------
        frame_width : int
            Width of the video frames
        frame_height : int
            Height of the video frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Tracking parameters
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object information
        self.max_disappeared = 250  # Maximum number of frames an object can be missing
        self.max_boundary_disappeared = 5  # Faster disappearance for objects near boundaries
        self.max_distance = 100  # Maximum distance to consider objects as the same

        # Trajectory parameters
        self.max_trajectory_points = 30  # Maximum number of points to keep in trajectory for display
        self.trajectories = {}  # Dictionary to store trajectory points
        self.full_trajectories = defaultdict(list)  # Full trajectory history

        # Color management
        self.object_colors = {}  # Dictionary to store consistent colors for each object ID

        # Frame counter
        self.frame_count = 0

    def set_frame_count(self, frame_count):
        """Set the current frame count."""
        self.frame_count = frame_count

    def track_objects(self, detections, frame):
        """
        Track objects across frames and assign consistent IDs.

        Parameters:
        -----------
        detections : list
            List of detection dictionaries
        frame : numpy.ndarray
            Current frame

        Returns:
        --------
        dict
            Dictionary of tracked objects with their IDs
        """
        # If no objects are being tracked yet, initialize tracking
        if len(self.objects) == 0:
            for detection in detections:
                x, y, w, h = detection['bbox']
                is_near_boundary = self.is_near_boundary(x, y, w, h)
                self.objects[self.next_object_id] = {
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'disappeared': 0,
                    'boundary_disappeared': 0,
                    'near_boundary': is_near_boundary,
                    'last_visible_bbox': detection['bbox'],
                    'last_visible_center': detection['center']
                }
                # Generate color for new object
                self.get_object_color(self.next_object_id)
                self.next_object_id += 1
            return self.objects

        # If no detections in current frame, handle all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                if self.objects[object_id]['near_boundary']:
                    self.objects[object_id]['boundary_disappeared'] += 1
                    if self.objects[object_id]['boundary_disappeared'] >= self.max_boundary_disappeared:
                        del self.objects[object_id]
                else:
                    self.objects[object_id]['disappeared'] += 1
                    if self.objects[object_id]['disappeared'] >= self.max_disappeared:
                        del self.objects[object_id]
            return self.objects

        # Calculate distances between existing objects and new detections
        object_ids = [obj_id for obj_id in self.objects.keys()]
        object_centers = [self.objects[obj_id]['center'] for obj_id in object_ids]
        detection_centers = [det['center'] for det in detections]

        # If we have existing objects, try to match them with new detections
        if len(object_centers) > 0:
            # Calculate distances between all pairs of existing objects and new detections
            distances = []
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distance = np.sqrt((obj_center[0] - det_center[0])**2 +
                                     (obj_center[1] - det_center[1])**2)
                    distances.append((i, j, distance))

            # Sort distances
            distances.sort(key=lambda x: x[2])

            # Match objects with detections
            matched_objects = set()
            matched_detections = set()
            for i, j, distance in distances:
                if distance > self.max_distance:
                    continue
                if i not in matched_objects and j not in matched_detections:
                    x, y, w, h = detections[j]['bbox']
                    is_near_boundary = self.is_near_boundary(x, y, w, h)

                    # Update object state
                    self.objects[object_ids[i]].update({
                        'center': detection_centers[j],
                        'bbox': detections[j]['bbox'],
                        'near_boundary': is_near_boundary,
                        'last_visible_bbox': detections[j]['bbox'],
                        'last_visible_center': detection_centers[j]
                    })

                    # Reset appropriate counters
                    if is_near_boundary:
                        self.objects[object_ids[i]]['boundary_disappeared'] = 0
                    else:
                        self.objects[object_ids[i]]['disappeared'] = 0
                        self.objects[object_ids[i]]['boundary_disappeared'] = 0

                    matched_objects.add(i)
                    matched_detections.add(j)

            # Handle unmatched objects
            for i in range(len(object_centers)):
                if i not in matched_objects:
                    obj_id = object_ids[i]

                    if self.objects[obj_id]['near_boundary']:
                        self.objects[obj_id]['boundary_disappeared'] += 1
                        if self.objects[obj_id]['boundary_disappeared'] > self.max_boundary_disappeared:
                            del self.objects[obj_id]
                    else:
                        self.objects[obj_id]['disappeared'] += 1
                        if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                            del self.objects[obj_id]

            # Handle new detections
            for j in range(len(detection_centers)):
                if j not in matched_detections:
                    x, y, w, h = detections[j]['bbox']
                    center_x, center_y = detection_centers[j]
                    is_near_boundary = self.is_near_boundary(x, y, w, h)
                    # Regular new detection (not near occlusion)
                    self.objects[self.next_object_id] = {
                        'center': detection_centers[j],
                        'bbox': detections[j]['bbox'],
                        'disappeared': 0,
                        'boundary_disappeared': 0,
                        'near_boundary': is_near_boundary,
                        'last_visible_bbox': detections[j]['bbox'],
                        'last_visible_center': detection_centers[j]
                    }
                    # Generate color for new object
                    self.get_object_color(self.next_object_id)
                    self.next_object_id += 1

        return self.objects

    def is_near_boundary(self, x, y, w, h):
        """
        Check if object is near the frame boundary.

        Parameters:
        -----------
        x, y, w, h : int
            Bounding box coordinates and dimensions

        Returns:
        --------
        bool
            True if object is near boundary, False otherwise
        """
        margin = 20  # pixels from boundary
        return (x <= margin or
                y <= margin or
                x + w >= self.frame_width - margin or
                y + h >= self.frame_height - margin)

    def update_trajectories(self, objects):
        """
        Update trajectories for all tracked objects.

        Parameters:
        -----------
        objects : dict
            Dictionary of tracked objects
        """
        for object_id, obj in objects.items():
            # Initialize trajectory if it doesn't exist
            if object_id not in self.trajectories:
                self.trajectories[object_id] = []
                self.full_trajectories[object_id] = []

            # Add current position to trajectory
            self.trajectories[object_id].append(obj['center'])
            self.full_trajectories[object_id].append(obj['center'])

        # Remove trajectories for objects that no longer exist
        for object_id in list(self.trajectories.keys()):
            if object_id not in objects:
                del self.trajectories[object_id]

    def update_trajectories_with_kalman(self, objects, kalman_manager):
        """
        Update trajectories with Kalman-filtered positions if available.

        Parameters:
        -----------
        objects : dict
            Dictionary of tracked objects
        kalman_manager : KalmanFilterManager
            Kalman filter manager instance
        """
        for object_id, obj in objects.items():
            # Get position - either Kalman-filtered or raw detection
            kalman_center = kalman_manager.get_kalman_center(object_id)
            if kalman_manager.use_kalman and kalman_center:
                position = kalman_center
            else:
                position = obj['center']

            # Create frame data matching readFiles.py structure
            x, y = position
            x1, y1, w, h = obj['bbox']

            frame_data = {
                'frame_number': self.frame_count,
                'position': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x1 + w),
                    'y2': float(y1 + h)
                },
                'metadata': [0, 0, 0],  # Default metadata
                'class': 'Unknown'  # Default class
            }

            # Add to full trajectories
            self.full_trajectories[object_id].append(frame_data)

            # Add position to real-time trajectory
            if object_id not in self.trajectories:
                self.trajectories[object_id] = []
            self.trajectories[object_id].append((x, y))  # Store as tuple of (x,y)

            # Limit trajectory points for real-time visualization
            if len(self.trajectories[object_id]) > self.max_trajectory_points:
                self.trajectories[object_id] = self.trajectories[object_id][-self.max_trajectory_points:]

    def get_object_color(self, object_id):
        """
        Get a random but consistent color for an object ID.

        Parameters:
        -----------
        object_id : int
            Object ID

        Returns:
        --------
        tuple
            BGR color tuple
        """
        if object_id not in self.object_colors:
            # Generate random BGR values, but ensure at least one component is bright
            while True:
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)

                # Ensure at least one component is bright (above 200)
                if max(b, g, r) > 200:
                    break

            self.object_colors[object_id] = (b, g, r)

        return self.object_colors[object_id]

    def ensure_colors_for_all_objects(self):
        """Ensure all tracked objects have colors assigned."""
        for object_id in self.objects.keys():
            if object_id not in self.object_colors:
                self.get_object_color(object_id)
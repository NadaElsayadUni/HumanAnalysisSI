"""
Visualization Module

This module handles all visualization and drawing operations for trajectories,
detections, and analysis results.
"""

import cv2
import numpy as np
import os


class Visualizer:
    """
    Handles visualization and drawing operations for the analysis results.
    """

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def draw_trajectories(self, frame, trajectories, object_colors, frame_width, frame_height):
        """
        Draw trajectories on the frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            Frame to draw on
        trajectories : dict
            Dictionary of trajectories for each object
        object_colors : dict
            Dictionary of colors for each object
        frame_width : int
            Width of the frame
        frame_height : int
            Height of the frame
        """
        for object_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            if object_id in object_colors:
                color = object_colors[object_id]
            else:
                color = (255, 255, 255)  # Default white if not found

            # Draw trajectory line
            for i in range(1, len(trajectory)):
                start_point = tuple(map(int, trajectory[i-1]))
                end_point = tuple(map(int, trajectory[i]))

                # Only draw if points are within frame
                if (0 <= start_point[0] < frame_width and
                    0 <= start_point[1] < frame_height and
                    0 <= end_point[0] < frame_width and
                    0 <= end_point[1] < frame_height):
                    thickness = 2
                    cv2.line(frame, start_point, end_point, color, thickness)

            # Draw current position
            if trajectory:
                current_point = tuple(map(int, trajectory[-1]))
                if (0 <= current_point[0] < frame_width and
                    0 <= current_point[1] < frame_height):
                    cv2.circle(frame, current_point, 4, color, -1)

    def draw_complete_trajectories_realtime(self, frame, full_trajectories, tracked_objects,
                                          object_colors, frame_width, frame_height):
        """
        Draw complete trajectories and detections during video playback (real-time).

        Parameters:
        -----------
        frame : numpy.ndarray
            Frame to draw on
        full_trajectories : dict
            Dictionary of full trajectories for each object
        tracked_objects : dict
            Dictionary of currently tracked objects
        object_colors : dict
            Dictionary of colors for each object
        frame_width : int
            Width of the frame
        frame_height : int
            Height of the frame
        """


        # Draw complete trajectories
        for object_id, trajectory in full_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            if object_id in object_colors:
                color = object_colors[object_id]
            else:
                color = (255, 255, 255)  # Default white if not found

            # Extract center points from trajectory
            x_coords = []
            y_coords = []
            for frame_data in trajectory:
                if isinstance(frame_data, dict) and 'position' in frame_data:
                    # Handle dictionary format (from update_trajectories_with_kalman)
                    x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                    x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                else:
                    # Handle tuple format (from update_trajectories)
                    center_x, center_y = frame_data
                x_coords.append(center_x)
                y_coords.append(center_y)

            # Draw current position (last point)
            if x_coords and y_coords:
                current_point = (int(x_coords[-1]), int(y_coords[-1]))
                if (0 <= current_point[0] < frame_width and
                    0 <= current_point[1] < frame_height):
                    cv2.circle(frame, current_point, 4, color, -1)

        # Draw detections and IDs if tracked_objects are provided
        if tracked_objects is not None:
            for object_id, obj in tracked_objects.items():
                x, y, w, h = obj['bbox']
                center_x, center_y = obj['center']

                # Get consistent color for this object
                if object_id in object_colors:
                    color = object_colors[object_id]
                else:
                    color = (255, 255, 255)  # Default white if not found
                border_color = color
                border_thickness = 2

                # Draw bounding box (allowing it to extend beyond frame boundaries)
                # Calculate visible portion of the box
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_width, x + w)
                y2 = min(frame_height, y + h)

                # Draw the visible portion of the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

                # Draw center point (if it's within the frame)
                if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
                    cv2.circle(frame, (center_x, center_y), 4, color, -1)

                # Draw ID (if the top of the box is visible)
                if y1 < frame_height:
                    id_text = f"ID: {object_id}"
                    cv2.putText(frame, id_text, (x1, max(10, y1 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def smooth_trajectory(self, x, y, window_size=5):
        """
        Smooth trajectory points using convolution.

        Parameters:
        -----------
        x : list
            X coordinates
        y : list
            Y coordinates
        window_size : int
            Size of smoothing window

        Returns:
        --------
        tuple
            (x_smooth, y_smooth) smoothed coordinates
        """
        if len(x) < window_size:
            return x, y  # Not enough points to smooth

        x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

        # Pad to match original length (center alignment)
        pad = (len(x) - len(x_smooth)) // 2
        x_smooth = np.pad(x_smooth, (pad, len(x) - len(x_smooth) - pad), mode='edge')
        y_smooth = np.pad(y_smooth, (pad, len(y) - len(y_smooth) - pad), mode='edge')

        return x_smooth, y_smooth

    def draw_complete_trajectories(self, last_frame, full_trajectories, object_colors,
                                 kalman_centers=None, use_kalman=False):
        """
        Draw complete trajectories on the given frame or reference frame.

        Parameters:
        -----------
        last_frame : numpy.ndarray
            Frame to draw on
        full_trajectories : dict
            Dictionary of full trajectories for each object
        object_colors : dict
            Dictionary of colors for each object
        kalman_centers : dict
            Dictionary of Kalman-filtered centers
        use_kalman : bool
            Whether to use Kalman-filtered positions

        Returns:
        --------
        numpy.ndarray or None
            Frame with trajectories drawn, or None if no frame available
        """
        # Use the provided last frame if available, otherwise use reference frame
        if last_frame is not None:
            trajectory_image = last_frame.copy()
        else:
            print("No frame available for drawing trajectories")
            return None

        # Create a legend image to show object IDs and their colors
        legend_height = min(200, len(full_trajectories) * 25)
        legend_width = 200
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        # Sort object IDs for consistent legend
        sorted_object_ids = sorted(full_trajectories.keys())

        # Draw all trajectories
        for i, object_id in enumerate(sorted_object_ids):
            trajectory = full_trajectories[object_id]
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            if object_id in object_colors:
                color = object_colors[object_id]
            else:
                color = (255, 255, 255)  # Default white if not found

            if use_kalman and kalman_centers and object_id in kalman_centers:
                # Draw only Kalman-filtered position
                kalman_x, kalman_y = kalman_centers[object_id]
                cv2.circle(trajectory_image, (int(kalman_x), int(kalman_y)), 4, color, -1)
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (int(kalman_x) + 10, int(kalman_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Draw raw trajectory
                x_coords = []
                y_coords = []
                for frame_data in trajectory:
                    if isinstance(frame_data, dict) and 'position' in frame_data:
                        # Handle dictionary format
                        x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                        x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    else:
                        # Handle tuple format
                        center_x, center_y = frame_data
                    x_coords.append(center_x)
                    y_coords.append(center_y)

                # Smooth trajectory points
                x_smooth, y_smooth = self.smooth_trajectory(x_coords, y_coords)
                pts = np.array(list(zip(x_smooth, y_smooth)), dtype=np.int32)

                if pts.shape[0] > 1:
                    cv2.polylines(trajectory_image, [pts], False, color, 2)

                # Draw start and end points
                start_point = (int(x_coords[0]), int(y_coords[0]))
                end_point = (int(x_coords[-1]), int(y_coords[-1]))

                # Draw start point as a circle
                cv2.circle(trajectory_image, start_point, 8, color, -1)
                cv2.circle(trajectory_image, start_point, 8, (255, 255, 255), 2)

                # Draw end point as a square
                square_size = 8
                cv2.rectangle(trajectory_image,
                             (end_point[0]-square_size, end_point[1]-square_size),
                             (end_point[0]+square_size, end_point[1]+square_size),
                             color, -1)
                cv2.rectangle(trajectory_image,
                             (end_point[0]-square_size, end_point[1]-square_size),
                             (end_point[0]+square_size, end_point[1]+square_size),
                             (255, 255, 255), 2)

                # Draw object ID at the end of trajectory
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (end_point[0] + 10, end_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (end_point[0] + 10, end_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Add to legend if it fits
            if i < legend_height // 25:
                y_pos = i * 25 + 15
                # Draw color sample
                cv2.rectangle(legend, (10, y_pos-10), (30, y_pos+10), color, -1)
                # Draw ID text
                cv2.putText(legend, f"ID: {object_id}", (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add legend to the right side of the trajectory image if there's space
        if trajectory_image.shape[1] > 800:
            x_offset = trajectory_image.shape[1] - legend_width - 10
            y_offset = 10
            if legend_height + y_offset < trajectory_image.shape[0]:
                trajectory_image[y_offset:y_offset+legend_height, x_offset:x_offset+legend_width] = legend

        return trajectory_image

    def save_individual_trajectories(self, last_frame, full_trajectories, object_colors, output_dir="detected_trajectories"):
        """
        Save individual trajectory images for each object.

        Parameters:
        -----------
        last_frame : numpy.ndarray
            Last frame to use as background
        full_trajectories : dict
            Dictionary of full trajectories for each object
        object_colors : dict
            Dictionary of colors for each object
        output_dir : str
            Output directory for trajectory images
        """
        if last_frame is None:
            print("No frame available for drawing individual trajectories")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each object's trajectory
        for object_id, trajectory in full_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Create a copy of the last frame
            trajectory_image = last_frame.copy()

            # Get color for this object
            if object_id in object_colors:
                color = object_colors[object_id]
            else:
                color = (255, 255, 255)  # Default white if not found

            # Extract center points from trajectory
            x_coords = []
            y_coords = []
            for frame_data in trajectory:
                if isinstance(frame_data, dict) and 'position' in frame_data:
                    x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                    x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                else:
                    center_x, center_y = frame_data
                x_coords.append(center_x)
                y_coords.append(center_y)

            # Convert to numpy array for drawing
            pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)

            # Draw the trajectory
            if pts.shape[0] > 1:
                cv2.polylines(trajectory_image, [pts], False, color, 3)

            # Draw start and end points
            start_point = (int(x_coords[0]), int(y_coords[0]))
            end_point = (int(x_coords[-1]), int(y_coords[-1]))

            # Draw start point as a circle
            cv2.circle(trajectory_image, start_point, 8, color, -1)
            cv2.circle(trajectory_image, start_point, 8, (255, 255, 255), 2)

            # Draw end point as a square
            square_size = 8
            cv2.rectangle(trajectory_image,
                         (end_point[0]-square_size, end_point[1]-square_size),
                         (end_point[0]+square_size, end_point[1]+square_size),
                         color, -1)
            cv2.rectangle(trajectory_image,
                         (end_point[0]-square_size, end_point[1]-square_size),
                         (end_point[0]+square_size, end_point[1]+square_size),
                         (255, 255, 255), 2)

            # Add object ID and trajectory information
            cv2.putText(trajectory_image, f"Object ID: {object_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(trajectory_image, f"Trajectory Points: {len(trajectory)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save the image
            output_path = os.path.join(output_dir, f"trajectory_object_{object_id}.jpg")
            cv2.imwrite(output_path, trajectory_image)
            print(f"Saved trajectory image for object {object_id} to {output_path}")
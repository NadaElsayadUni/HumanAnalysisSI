"""
Main Analyzer Module

This module contains the HumanMotionAnalyzer class that orchestrates
all the analysis components including detection, tracking, Kalman filtering,
and trajectory analysis.
"""

import cv2
import numpy as np
from pathlib import Path

from .detection import ObjectDetector
from .tracking import ObjectTracker
from .kalman_filter import KalmanFilterManager
from ..utils.visualization import Visualizer


class HumanMotionAnalyzer:
    """
    Main analyzer class for human motion analysis in videos.

    This class orchestrates all the analysis components including object detection,
    tracking, Kalman filtering, and trajectory analysis.
    """

    def __init__(self, video_path, reference_path):
        """
        Initialize the human motion analyzer.

        Parameters:
        -----------
        video_path : str
            Path to the input video file
        reference_path : str
            Path to the reference/background image
        """
        self.video_path = video_path
        self.reference_path = reference_path

        # Video and frame management
        self.cap = None
        self.reference_frame = None
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None

        # Initialize analysis components
        self.detector = ObjectDetector()
        self.tracker = None
        self.kalman_manager = KalmanFilterManager()
        self.visualizer = Visualizer()

    def load_reference_frame(self):
        """
        Load the reference frame (background).

        Returns:
        --------
        numpy.ndarray
            Loaded reference frame
        """
        self.reference_frame = cv2.imread(self.reference_path)
        if self.reference_frame is None:
            raise ValueError(f"Could not load reference frame from {self.reference_path}")
        return self.reference_frame

    def load_video(self):
        """
        Load the video file.

        Returns:
        --------
        cv2.VideoCapture
            Video capture object
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file {self.video_path}")

        # Get video dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo resolution: {self.frame_width}x{self.frame_height}")

        # Initialize tracker with video dimensions
        self.tracker = ObjectTracker(self.frame_width, self.frame_height)

        return self.cap

    def process_frame(self, frame):
        """
        Process a single frame through the complete analysis pipeline.

        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to process

        Returns:
        --------
        dict
            Dictionary containing processing results
        """
        # Update frame counter
        self.frame_count += 1
        self.tracker.set_frame_count(self.frame_count)

        # Detect moving objects
        detections, fg_mask, shadow_mask = self.detector.detect_moving_objects(frame)

        # Track objects
        tracked_objects = self.tracker.track_objects(detections, frame)

        # Update Kalman filters
        if self.kalman_manager.use_kalman:
            self.kalman_manager.update_kalman_filters(tracked_objects)
            # Use Kalman-filtered trajectories
            self.tracker.update_trajectories_with_kalman(tracked_objects, self.kalman_manager)
        else:
            # Use regular trajectories
            self.tracker.update_trajectories(tracked_objects)

        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'fg_mask': fg_mask,
            'shadow_mask': shadow_mask,
            'frame_count': self.frame_count
        }

    def run_analysis(self, show_visualization=True, save_results=True):
        """
        Run the complete analysis pipeline on the video.

        Parameters:
        -----------
        show_visualization : bool
            Whether to show real-time visualization
        save_results : bool
            Whether to save results and visualizations
        """
        # Load video and reference
        self.load_reference_frame()
        self.load_video()

        # Process video frames
        frame_count = 0
        last_frame = None
        last_shadow_mask = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video reached after {frame_count} frames")
                break

            # Store the current frame as the last frame
            last_frame = frame.copy()

            # Process frame
            results = self.process_frame(frame)

            # Store shadow mask from last frame
            last_shadow_mask = results['shadow_mask']

            # Ensure all tracked objects have colors assigned
            self.tracker.ensure_colors_for_all_objects()

            # Draw results
            self.visualizer.draw_complete_trajectories_realtime(
                frame,
                self.tracker.full_trajectories,
                results['tracked_objects'],
                self.tracker.object_colors,
                self.frame_width,
                self.frame_height
            )

            # Display results
            if show_visualization:
                cv2.imshow('Analysis Results', frame)
                cv2.imshow('Foreground Mask', results['fg_mask'])
                cv2.imshow('Shadow Mask', results['shadow_mask'])

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User pressed 'q', exiting video processing")
                    break

            frame_count += 1

        # Clean up
        self.cap.release()

        # Save results if requested
        if save_results and last_frame is not None:
            self.save_analysis_results(last_frame, last_shadow_mask)

        if show_visualization:
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()

    def save_analysis_results(self, last_frame, last_shadow_mask=None):
        """
        Save analysis results including trajectories and visualizations.

        Parameters:
        -----------
        last_frame : numpy.ndarray
            Last frame to use for trajectory visualization
        last_shadow_mask : numpy.ndarray
            Shadow mask from the last frame
        """
        import os

        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Save last frame
        last_frame_path = os.path.join(output_dir, "last_frame_signal.jpg")
        cv2.imwrite(last_frame_path, last_frame)
        print(f"Last frame saved as {last_frame_path}")

        # Save shadow mask if available
        if last_shadow_mask is not None:
            shadow_mask_path = os.path.join(output_dir, "shadow_mask.jpg")
            cv2.imwrite(shadow_mask_path, last_shadow_mask)
            print(f"Shadow mask saved as {shadow_mask_path}")
            # Display shadow mask
            cv2.imshow('Shadow Mask', last_shadow_mask)

        # Draw and save raw trajectories
        raw_trajectory_image = self.visualizer.draw_complete_trajectories(
            last_frame,
            self.tracker.full_trajectories,
            self.tracker.object_colors,
            use_kalman=False
        )
        if raw_trajectory_image is not None:
            raw_path = os.path.join(output_dir, "trajectories_raw.jpg")
            cv2.imwrite(raw_path, raw_trajectory_image)
            print(f"Raw trajectories image saved as {raw_path}")
            # Display raw trajectories
            cv2.imshow('Raw Trajectories', raw_trajectory_image)

        # Draw and save Kalman-filtered trajectories
        if self.kalman_manager.use_kalman:
            kalman_trajectory_image = self.visualizer.draw_complete_trajectories(
                last_frame,
                self.tracker.full_trajectories,
                self.tracker.object_colors,
                self.kalman_manager.kalman_centers,
                use_kalman=True
            )
            if kalman_trajectory_image is not None:
                kalman_path = os.path.join(output_dir, "trajectories_kalman.jpg")
                cv2.imwrite(kalman_path, kalman_trajectory_image)
                print(f"Kalman-filtered trajectories saved as {kalman_path}")
                # Display Kalman trajectories
                cv2.imshow('Kalman-filtered Trajectories', kalman_trajectory_image)

        # Save individual trajectory images
        # print("Saving individual trajectory images...")
        # individual_trajectories_dir = os.path.join(output_dir, "detected_trajectories")
        # self.visualizer.save_individual_trajectories(
        #     last_frame,
        #     self.tracker.full_trajectories,
        #     self.tracker.object_colors,
        #     output_dir=individual_trajectories_dir
        # )
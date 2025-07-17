"""
Main script for Human Motion Analysis using Modular Structure

This script demonstrates how to use the modular human motion analysis system.
"""

from pathlib import Path
from src import HumanMotionAnalyzer


def main():
    """Main function to run the human motion analysis."""

    # Define paths
    reference_path = Path("video") / "reference.jpeg"
    video_path = Path("video") / "video.mp4"
    # Check if files exist
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        print("Please make sure the video file exists in the correct location.")
        return

    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        print("Please make sure the reference file exists in the correct location.")
        return

    print("=== Human Motion Analysis with Modular Structure ===")
    print(f"Video: {video_path}")
    print(f"Reference: {reference_path}")

    try:
        # Initialize the analyzer
        analyzer = HumanMotionAnalyzer(
            video_path=str(video_path),
            reference_path=str(reference_path)
        )

        # Run the analysis
        print("\nStarting analysis...")
        analyzer.run_analysis(
            show_visualization=True,  # Set to False to run without GUI
            save_results=True
        )

        print("\nAnalysis completed successfully!")
        print("\nOutput files generated:")
        print("- output/last_frame_signal.jpg: Last frame from video")
        print("- output/trajectories_raw.jpg: Raw trajectory visualization")
        print("- output/trajectories_kalman.jpg: Kalman-filtered trajectory visualization")
        print("- output/detected_trajectories/: Individual trajectory images for each object")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your video and reference files.")


if __name__ == "__main__":
    main()
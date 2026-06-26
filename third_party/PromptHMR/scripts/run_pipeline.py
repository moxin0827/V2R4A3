#!/usr/bin/env python3
"""
Run PromptHMR pipeline without visualization (viser)

Usage:
    python scripts/run_pipeline.py --input_video video.mp4
    python scripts/run_pipeline.py --input_video video.mp4 --static_camera
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import Pipeline


def main(
    input_video: str,
    static_camera: bool = False,
    output_folder: str = "results",
):
    """
    Run PromptHMR pipeline and save results (no visualization)
    
    Args:
        input_video: Path to input video
        static_camera: Assume static camera
        output_folder: Output folder for results
    """
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    print(f"[Pipeline] Input: {input_video}")
    print(f"[Pipeline] Static camera: {static_camera}")
    
    # Run pipeline
    pipeline = Pipeline(static_cam=static_camera)
    results = pipeline(
        input_video=str(input_video),
        output_folder=output_folder,
        static_cam=static_camera,
    )
    
    # Results are saved automatically by pipeline
    # PromptHMR saves to output_folder directly (not output_folder/video_name)
    results_dir = Path(output_folder)
    
    print(f"\n[Pipeline] Complete!")
    print(f"[Pipeline] Results saved to: {results_dir}")
    print(f"[Pipeline] Files:")
    for f in sorted(results_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PromptHMR pipeline (no visualization)")
    parser.add_argument("--input_video", "-i", required=True, help="Input video path")
    parser.add_argument("--static_camera", "-s", action="store_true", help="Assume static camera")
    parser.add_argument("--output_folder", "-o", default="results", help="Output folder")
    
    args = parser.parse_args()
    
    main(
        input_video=args.input_video,
        static_camera=args.static_camera,
        output_folder=args.output_folder,
    )


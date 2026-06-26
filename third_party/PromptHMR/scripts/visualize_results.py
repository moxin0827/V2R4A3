#!/usr/bin/env python3
"""
Visualize PromptHMR results with viser (floor, camera, etc.)

Usage:
    python scripts/visualize_results.py --results_dir results/original
"""

import os
import sys
import argparse
import time
import joblib
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh


def load_video_frames(video_path, max_frames=None):
    """Load frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def main(
    results_dir: str,
    video_path: str = None,
    viser_total: int = 1500,
    viser_subsample: int = 1,
):
    """
    Visualize results with viser
    
    Args:
        results_dir: Path to results directory (containing results.pkl)
        video_path: Path to video file (for camera images)
        viser_total: Max frames to visualize
        viser_subsample: Subsample rate
    """
# Load results
    results_pkl = os.path.join(results_dir, "results.pkl")
    if not os.path.exists(results_pkl):
        raise FileNotFoundError(f"results.pkl not found: {results_pkl}")
    
    print(f"[Visualize] Loading: {results_pkl}")
    results = joblib.load(results_pkl)

    # Load SMPLX
    print(f"[Visualize] Loading SMPLX model...")
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    
    # Try to find video
    if video_path is None:
        # Try common locations
        for name in ["original.mp4", "video.mp4"]:
            test_path = os.path.join(os.path.dirname(results_dir), name)
            if os.path.exists(test_path):
                video_path = test_path
                break
    
    # Load video frames if available
    if video_path and os.path.exists(video_path):
        print(f"[Visualize] Loading video: {video_path}")
        images = load_video_frames(video_path, max_frames=viser_total)
        images = images[::viser_subsample]
    else:
        print(f"[Visualize] No video found, using placeholder images")
        # Create placeholder images
        num_frames = len(results.get("people", {}).get(list(results.get("people", {}).keys())[0], {}).get("frames", []))
        images = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(min(num_frames, viser_total))]
        images = images[::viser_subsample]
    
    # Create world4d from results
    print(f"[Visualize] Creating world4d...")
    world4d = create_world4d_from_results(results, step=viser_subsample, total=viser_total)
    
    # Compute vertices
    print(f"[Visualize] Computing vertices...")
    all_verts = []
    for k in world4d:
        world3d = world4d[k]
        if len(world3d['track_id']) == 0:
            continue
        
        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        verts = smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=world3d['shape'].cuda(),
            transl=world3d['trans'].cuda()
        ).vertices.cpu().numpy()
        
        world3d['vertices'] = verts
        all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))
    
    # Create floor mesh
    if all_verts:
        all_verts = torch.cat(all_verts)
        [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)
        floor = [gv, gf]
    else:
        floor = None
    
    # Start viser
    print(f"[Visualize] Starting viser...")
    server, gui = viser_vis_world4d(
        images,
        world4d,
        smplx.faces,
        floor=floor,
        init_fps=30 / viser_subsample
    )
    
    url = f'https://localhost:{server.get_port()}'
    print(f"\n[Visualize] Open in browser: {url}")
    print(f"[Visualize] Press Ctrl+C to stop")

    gui_playing, gui_timestep, gui_framerate, num_frames = gui
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


def create_world4d_from_results(results, step=1, total=None):
    """Create world4d dict from PromptHMR results"""
    camera = results.get('camera_world', results.get('camera', {}))
    
    if 'Rwc' in camera:
        num_frames = len(camera['Rwc'])
    else:
        # Fallback
        people = results.get('people', {})
        if people:
            first_person = list(people.values())[0]
            num_frames = len(first_person.get('frames', []))
        else:
            num_frames = 0
    
    if total:
        num_frames = min(num_frames, total)
    
    world4d = {}
    for i in range(0, num_frames, step):
        pose = []
        shape = []
        transl = []
        track_id = []
        
        # People
        for pid, person in results.get('people', {}).items():
            frames = person.get('frames', np.arange(100))
            in_frame = np.where(frames == i)[0]
            
            if len(in_frame) == 1:
                smplx_w = person.get('smplx_world', {})
                if smplx_w:
                    pose.append(smplx_w['pose'][in_frame])
                    shape.append(smplx_w['shape'][in_frame])
                    transl.append(smplx_w['trans'][in_frame])
                    track_id.append(person.get('track_id', pid))
        
        # Camera
        if 'Rwc' in camera:
            Rwc = camera['Rwc'][i]
            Twc = camera['Twc'][i]
        else:
            Rwc = np.eye(3)
            Twc = np.zeros(3)
        
        cam_matrix = np.eye(4)
        cam_matrix[:3, :3] = Rwc
        cam_matrix[:3, 3] = Twc
        
        idx = i // step
        if len(track_id) > 0:
            world4d[idx] = {
                'pose': torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3),
                'shape': torch.tensor(np.concatenate(shape)).float(),
                'trans': torch.tensor(np.concatenate(transl)).float(),
                'track_id': torch.tensor(np.array(track_id)) - 1,
                'camera': cam_matrix
            }
        else:
            world4d[idx] = {
                'track_id': np.array([]),
                'camera': cam_matrix
            }
    
    return world4d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PromptHMR results")
    parser.add_argument("--results_dir", "-r", required=True, help="Results directory")
    parser.add_argument("--video", "-v", help="Video file path")
    parser.add_argument("--total", type=int, default=1500, help="Max frames")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate")
    
    args = parser.parse_args()
    
    main(
        results_dir=args.results_dir,
        video_path=args.video,
        viser_total=args.total,
        viser_subsample=args.subsample,
    )

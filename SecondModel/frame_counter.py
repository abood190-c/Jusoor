import os

import numpy as np
import cv2

videos_dir = os.path.join('Dataset', 'start_kit', 'videos')

def video_to_frames(videos_dir, size=None):
    max_frame_count = 0
    min_frame_count = float('inf')
    avg_frame_count = 0
    video_count = 0
    high_frame_videos_count = 0
    low_frame_videos_count = 0

    for filename in os.listdir(videos_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(videos_dir, filename)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame_count = max(max_frame_count, frame_count)
            min_frame_count = min(min_frame_count, frame_count)
            avg_frame_count += frame_count
            video_count += 1
            if frame_count > 150:
                high_frame_videos_count += 1
            if frame_count < 30:
                low_frame_videos_count += 1
            print(video_count)
            cap.release()
    
    if video_count > 0:
        avg_frame_count /= video_count
    
    print(f'Max frame count: {max_frame_count}')
    print(f'Min frame count: {min_frame_count}')
    print(f'Average frame count: {avg_frame_count}')
    print(f'Videos with more than 150 frames: {high_frame_videos_count}')
    print(f'Videos with less than 30 frames: {low_frame_videos_count}')

if __name__ == "__main__":
    video_to_frames(videos_dir)
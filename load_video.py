import os 
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2

def video_capture(video_path):
    video_cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video_cap.read()
    while success:
        frames.append(frame)
        success, frame = video_cap.read()
    
    return frames, len(frames)


def load_video(video_path, transform=None, max_size=None):
    if os.path.exists(video_path):
        if video_path.endswith('mp4') or video_path.endswith('avi') or video_path.endswith('gif'):
            frames, num_frames = video_capture(video_path=video_path)
            print(f'Number of frames: {num_frames}')
        else:
            raise ValueError('{} is not a valid video file'.format(video_path))
    else:
        raise ValueError('{} does not exist'.format(video_path))
    
    
    for i, frame in enumerate(frames):
        if max_size:
            if isinstance (frame, Image.Image):
                scale = max_size / max(frame.size)
                size = np.array(frame.size) * scale
                frame = frame.resize(size.astype(int), Image.ANTIALIAS)
            else:
                # Assuming frame is an OpenCV image (NumPy array)
                print(frame.shape)
                if frame.shape == 3:
                    height, width = frame.shape[:2]
                    scale = max_size / max(height, width)
                    new_size = (int(width * scale), int(height * scale))
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                    
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
        if transform:
            frame = transform(frame)
            
        frames[i] = frame
    
    return frames
        
        
# class VideoDataset(Dataset):
#     def __init__(self, video_path, transform=None, max_size=None):
#         self.video_path = video_path
#         self.transform = transform
#         self.max_size = max_size
#         self.frames = load_video(video_path, transform, max_size)
#         self.num_frames = len(self.frames)
#         print(f"Loaded {self.num_frames} frames from {self.video_path}")
        
#     def __len__(self):
#         return self.num_frames
    
#     def __getitem__(self, idx):
#         frame = self.frames[idx]
#         frame_tensor = transforms.ToTensor()(frame)  # Ensure frame is converted to tensor
#         return frame_tensor


class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None, max_size=None):
        self.video_path = video_path
        self.transform = transform
        self.max_size = max_size
        self.frames, self.num_frames = self.load_video()
        
    def load_video(self):
        video_cap = cv2.VideoCapture(self.video_path)
        frames = []
        success, frame = video_cap.read()
        while success:
            frames.append(frame)
            success, frame = video_cap.read()
        return frames, len(frames)
    
    def resize_frame(self, frame):
        height, width = frame.shape[:2]
        scale = self.max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.max_size:
            frame = self.resize_frame(frame)
        if self.transform and isinstance(frame, Image.Image):
            frame = self.transform(frame)
        elif self.transform and isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
        return frame


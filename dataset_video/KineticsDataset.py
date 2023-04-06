import torch.utils.data as data
import os
import glob
from tqdm import tqdm 
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu
import json
import lmdb 
from io import BytesIO
import time
import msgpack 

'''
K400Dataset for Pre-Training
'''
class KineticsDataset(data.Dataset):
    def __init__(self, transform, frame_gap, root, check_corrupted_video = False):
        self.root = root
        self.transform = transform
        self.frame_gap = frame_gap

        # if there is corrupted videos
        if check_corrupted_video:
            video_corrupted_list = json.load(open('/apdcephfs/private_qiangqwu/Projects/mae/corrputed_videos.json'))

        action_names = os.listdir(self.root)
        self.video_list = []
        for i in tqdm(range(len(action_names))):
            action_name = action_names[i]
            video_names = glob.glob(os.path.join(self.root, action_name, '*.mp4'))
            for video_name in video_names:
                if check_corrupted_video:
                    '''
                    check whether the downloaded K400 video is corrupted
                    '''
                    if os.path.join('/apdcephfs/share_1290939/0_public_datasets/k400/data/videos/train', action_name, video_name.split('/')[-1]) not in video_corrupted_list:
                        self.video_list.append(video_name)
                    else:
                        print('skip corrupted ' + video_name)
                else:
                    self.video_list.append(video_name)

        print('number of videos: %d' %(len(self.video_list)))
        self.total_size = len(self.video_list)
        print('frame_gap: %d' %(self.frame_gap))
        

    def __getitem__(self, index):
        while True:
            index = index % len(self.video_list)
            video_path = self.video_list[index]
            try:
                # decord the video 
                frames = VideoReader(video_path, num_threads=1, ctx=cpu(0))
                if len(frames) == 0:
                    index += 1
                    continue
                break 
            except:
                # catch the exception
                index += 1
                print(video_path)
        if len(frames) == 1:
            frame_index_list = [0, 0] # repeat the frame
        else:
            frame_index_list = [i for i in range(len(frames))]
        start_idx = np.random.randint(0, len(frame_index_list)-1) # values[0, len-1-1]
        end_idx = np.random.randint(start_idx+1, min(len(frame_index_list), start_idx + self.frame_gap))

        img_x = frames[frame_index_list[start_idx]].asnumpy()
        img_z = frames[frame_index_list[end_idx]].asnumpy()

        img_x = Image.fromarray(img_x, 'RGB')
        img_z = Image.fromarray(img_z, 'RGB')

        return self.transform(img_x), self.transform(img_z)
    
    def __len__(self):
        return self.total_size

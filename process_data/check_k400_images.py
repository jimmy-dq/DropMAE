import glob
import os
from decord import VideoReader
from tqdm import tqdm
import glob

# for frames
# count_videos = 0
# base_path = '/apdcephfs/share_1290939/0_public_datasets/k400/data/frames/train'
# classe_names = os.listdir(base_path)
# print(len(classe_names))
# for class_name in classe_names:
#     path = os.path.join(base_path, class_name)
#     count_videos += len(os.listdir(path))
# print(count_videos)


# for mp4 videos
# how many videos
# count_videos = 0
# base_path_2 = '/apdcephfs/share_1290939/0_public_datasets/k400/data/videos/train'
# classe_names = os.listdir(base_path_2)
# print(len(classe_names))
# for class_name in classe_names:
#     path = os.path.join(base_path_2, class_name)
#     count_videos += len(os.listdir(path))
# print(count_videos)

root = '/apdcephfs/share_1290939/0_public_datasets/k400/data/videos/train'
action_names = os.listdir(root)
video_list = []
for i in tqdm(range(len(action_names))):
            action_name = action_names[i]
            video_names = glob.glob(os.path.join(root, action_name, '*.mp4'))
            for video_name in video_names:
                try:
                    print(video_name)
                    frames = VideoReader(video_name)
                except:
                    print('exception')


'''
392
237159
400
241553
'''



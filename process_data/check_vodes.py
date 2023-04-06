import os
import os.path as osp
import glob
import threading


# 241553 training videos in K400
videos = glob.glob('/apdcephfs/share_1290939/0_public_datasets/k400/data/videos/train/*/*.mp4', recursive=True)
video_dict = {osp.splitext(osp.basename(video))[0][:-14] : video for video in videos}
k400_videos = open('/apdcephfs/share_1290939/0_public_datasets/k400/downloaded.txt').readlines()
k400_videos = [item.strip() for item in k400_videos]

# video_ids = list(set(video_dict.keys()) - set(k400_videos))
video_ids = list(video_dict.keys())
videos = [video_dict[video_id] for video_id in video_ids]

n_video = len(videos)
print('Total videos to check: {}'.format(n_video))

checks = [True] * n_video
checks_short = [True] * n_video


class myThread(threading.Thread):
    def __init__(self, threadId, name, ids, videos):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.ids = ids
        self.videos = videos

    def run(self):
        print('Begin thread, id: {}, name: {}'.format(self.threadId, self.name))
        cnt = 0
        for idx, video in zip(self.ids, self.videos):
            video_id = osp.splitext(osp.basename(video))[0][:-14]
            video_folder = '/tmp/video_temp/{}'.format(video_id)
            if not osp.exists(video_folder):
                os.makedirs(video_folder)
            try:
                os.system('ffmpeg -i {} /tmp/video_temp/{}/%d.jpg -loglevel panic'.format(video, video_id))
            except:
                print('Corrupted: {}'.format(video))
                checks[idx] = False
            if len(os.listdir(video_folder)) < 30:
                print('Corrupted (too short): {}'.format(video))
                checks_short[idx] = False
            os.system('rm -r {}'.format(video_folder))
            cnt += 1
            if cnt % 100 == 0:
                print('Thread {} processed {} files.'.format(self.threadId, cnt))


threads = []


n_thread = 6
part_size = n_video // n_thread
for i in range(n_thread):
    start_id = i * part_size
    if i == n_thread - 1:
        end_id = n_video
    else:
        end_id = (i+1)*part_size
    part_ids = list(range(start_id, end_id))
    part_videos = videos[start_id:end_id]
    mythread = myThread(i, 'thread-'+str(i), part_ids, part_videos)
    mythread.start()
    threads.append(mythread)

for t in threads:
    t.join()


valid_k600_videos = [video for check, video in zip(checks, videos) if check]
invalid_k600_videos = [video for check, video in zip(checks, videos) if not check]
with open('valid_videos.txt', 'w') as fid:
    fid.writelines([item+'\n' for item in valid_k600_videos])
with open('invalid_videos.txt', 'w') as fid:
    fid.writelines([item+'\n' for item in invalid_k600_videos])
with open('valid_video_ids.txt', 'w') as fid:
    fid.writelines([osp.splitext(osp.basename(item))[0][:-14]+'\n' for item in valid_k600_videos])
with open('invalid_video_ids.txt', 'w') as fid:
    fid.writelines([osp.splitext(osp.basename(item))[0][:-14]+'\n' for item in invalid_k600_videos])

valid_k600_videos = [video for check, video in zip(checks_short, videos) if check]
invalid_k600_videos = [video for check, video in zip(checks_short, videos) if not check]
with open('valid_videos_short.txt', 'w') as fid:
    fid.writelines([item+'\n' for item in valid_k600_videos])
with open('invalid_videos_short.txt', 'w') as fid:
    fid.writelines([item+'\n' for item in invalid_k600_videos])
with open('valid_video_ids_short.txt', 'w') as fid:
    fid.writelines([osp.splitext(osp.basename(item))[0][:-14]+'\n' for item in valid_k600_videos])
with open('invalid_video_ids_short.txt', 'w') as fid:
    fid.writelines([osp.splitext(osp.basename(item))[0][:-14]+'\n' for item in invalid_k600_videos])
import json
# a= []
# # read corructed videos
# f = open("/apdcephfs/private_qiangqwu/Projects/mae/corructed_videos.txt", 'r')
# for line in f.readlines():
#    line = line.strip()
#    if line in a:
#        continue
#    else:
#         a.append(line.strip())
# with open('/apdcephfs/private_qiangqwu/Projects/mae/corrputed_videos.json', 'w') as f:
#     json.dump(a, f)

list = json.load(open('/apdcephfs/private_qiangqwu/Projects/mae/corrputed_videos.json'))
print('a')

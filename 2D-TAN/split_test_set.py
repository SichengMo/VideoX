import os
import json

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))



if __name__ == '__main__':
    num_split = 16

    path_to_json =  os.path.join('data','MAD','mad.json')

    with open(path_to_json,'r') as f:
        data = json.load(f)
    test_videos = data['test']
    video_per_sub_set = len(list(test_videos.keys()))/num_split

    sub_lists = split(list(test_videos.keys()),num_split)
    count = 0
    sub_test_folder = os.path.join('data','MAD','sub_test')
    os.makedirs(sub_test_folder,exist_ok=True)

    for each in sub_lists:
        temp_content = dict()
        for video_id in each:
            temp_content[video_id] = test_videos[video_id]

        sub_test_name = os.path.join(sub_test_folder,('mad_test_{:02d}.json'.format(count)))
        #print(sub_test_name)
        count += 1
        temp_dict = {'test': temp_content}
        with open(sub_test_name,'w') as f:
            json.dump(temp_dict,f)
    print("Done!")



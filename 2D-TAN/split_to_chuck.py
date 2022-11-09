import os
import json

from tqdm import tqdm

import numpy as np


def iou(pred, gt):  # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    # print(overlap)
    return overlap


def get_highest_iou_window(windows, segment):
    """
    Get the intersection over union for a given window
    :param segment:
    :param window:
    :return:
    """
    # intersection = segment.intersection(window)
    # union = segment.union(window)
    # for window in windows:

    ious = iou(windows, segment)

    window_idx = np.argmax(ious)

    # print(window_idx)
    # print(windows[window_idx])
    return window_idx


def get_windows(num_frames, sub_chunk_size):
    """
    Get the windows for a given number of frames
    :param num_frames:
    :param sub_chunk_size:
    :return:
    """
    starts = np.arange(0, num_frames, sub_chunk_size)
    ends = starts + sub_chunk_size

    if ends[-1] > num_frames:
        ends[-1] = num_frames
        starts[-1] = ends[-1] - sub_chunk_size

    # print(starts)
    # print(ends)

    windows = []
    for idx in range(len(starts)):
        windows.append([starts[idx], ends[idx]])

    return windows


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, content):
    with open(path, 'w') as f:
        json.dump(content, f)

def create_chuck(window_length):
    num_split = 16
    #window_length = 3600
    sub_chunk_size = window_length * 5

    path_to_sub_test_orig = os.path.join('data', 'MAD', 'sub_test')
    video_path = os.path.join('data', 'MAD', '5ps_clip1_s1_512d')

    output_folder = os.path.join('data', 'MAD', 'chuck_{:04d}'.format(window_length))
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(num_split)):

        # print(i)
        sub_test_name = 'mad_test_{:02d}.json'.format(i)
        sub_test_path = os.path.join(path_to_sub_test_orig, sub_test_name)
        output_file_path = os.path.join(output_folder, sub_test_name)

        test_videos = load_json(sub_test_path)

        videos = test_videos['test']

        temp_all_new_videos = {}
        # get_highest_iou_window([[0,2], [0,3]], [1, 3])
        # get all windows
        for key in videos.keys():
            video = videos[key]
            windows = get_windows(video['num_frames'], sub_chunk_size)
            # for each in windows:
            #     print(each)

            if video['annotations']:
                annotations = video['annotations']

                for pair in annotations:
                    # print(pair)

                    sentence = pair['sentence']
                    segment = pair['segment']
                    sentence_id = pair['sentence_id']

                    window_idx = get_highest_iou_window(windows, segment)
                    # print(window_idx)
                    target_window = windows[window_idx]
                    target_window[0] = int(target_window[0])
                    target_window[1] = int(target_window[1])

                    start = max(target_window[0], segment[0]) - target_window[0]
                    end = min(target_window[1], segment[1]) - target_window[0]

                    start = int(start)
                    end = int(end)

                    segment = [start, end]

                    new_video_id = key + "_???_" + str(window_idx)

                    # print(new_video_id)

                    if new_video_id in temp_all_new_videos.keys():
                        temp_all_new_videos[new_video_id]['annotations'].append({
                            'segment': segment,
                            'sentence': sentence,
                            'sentence_id': sentence_id,
                        })
                    else:
                        new_video = {
                            'annotations': [
                                {
                                    'segment': segment,
                                    'sentence': sentence,
                                    'sentence_id': sentence_id,
                                }],
                            'num_frames': sub_chunk_size,
                            'duration': sub_chunk_size / 5,
                            'fps': 5,
                            'window': [target_window[0], target_window[1]],
                        }
                        temp_all_new_videos.update({new_video_id: new_video})
            # exit()
        new_test = {'test': temp_all_new_videos}
        # print(new_test)
        # exit()

        write_json(output_file_path, new_test)

if __name__ == '__main__':
    sizes = [30,60,120,180,600,1200,1800,3600]
    for size in sizes:
        create_chuck(size)

import math
import os
import json
import h5py
import logging
import random
import numpy as np
import pickle as pk
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from . import average_to_fixed_length
from .utils import movie2feats, moment_to_iou2d
from . import average_to_fixed_length
from core.eval import iou
from core.config import config
from tqdm import tqdm
from glob import glob


class MADOrigdataset(torch.utils.data.Dataset):

    # text centric dataset
    def __init__(self, split):

        self.sub_test_idx = config.TEST.SUB_TEST_INDEX
        #########################
        # cache video araa

        self.cache_videos = {}
        self.cache_texts = {}

        ##########################
        self.split = split
        data_dir = os.path.join('data', 'MAD')
        self.vid_feat_dir = os.path.join(data_dir, '5fps_clip1_s1_512d')
        self.text_feat_dir = os.path.join(data_dir, 'token_512d')
        data_orig_folder = os.path.join('data', 'MAD', 'orig')

        self.anno_file_path = {
            'train': os.path.join(data_orig_folder, 'MAD_train.json'),
            'val': os.path.join(data_orig_folder, "MAD_val.json"),
            "test": os.path.join(data_orig_folder, "MAD_test.json")

        }

        self.video_feat_all = os.path.join(data_orig_folder, 'CLIP_frames_features_5fps.h5')
        self.text_feat_all = os.path.join(data_orig_folder, 'CLIP_language_tokens_features-002.h5')

        if self.split == 'test' and self.sub_test_idx != -1:
            self.anno_file = os.path.join(data_dir, 'sub_test', 'mad_test_{:02d}.json'.format(self.sub_test_idx))
        else:
            self.anno_file = os.path.join(data_dir, 'mad.json')

        self.num_pre_clips = config.DATASET.NUM_SAMPLE_CLIPS
        self.test_stride = self.num_pre_clips / 2
        self.target_stride = config.DATASET.TARGET_STRIDE

        self.num_clips = int(self.num_pre_clips / self.target_stride)

        cache_file = os.path.join(data_dir, (split + '_orig.json'))

        index_file = os.path.join('data', 'MAD', ('mat_' + self.split + '_index.json'))

        # load cache anno file or create one
        try:
            with open(cache_file, 'r') as f:
                self.data_list = json.load(f)

            if self.split == 'test':
                self.query_interval_index = json.load(index_file)
                # use largest windows index as dataset size
                self.dataset_size = 0
                for key in self.query_interval_index.keys():
                    if self.query_interval_index(key)[1] >= self.dataset_size:
                        self.dataset_size = self.query_interval_index
                self.dataset_size = self.dataset_size - 1

            print("Loaded form cache file")

        except:
            print("Start loading")
            self.data_list = self._load_annotation()
            with open(cache_file, 'w') as f:
                json.dump(self.data_list, f)

        print(self.__len__())
        self.annotations = self.data_list
        self.movies = {a['id']: a['movie_duration'] for a in self.annotations}
        self.feats = movie2feats(self.video_feat_all, self.movies.keys())




        #self.text_feats = {}
        with h5py.File(self.text_feat_all, 'r') as f:
            self.text_feats = {a['sentence_id']: torch.from_numpy(f[a['sentence_id']][:].astype(np.float32)) for a in self.annotations}
            # for a in self.annotations:
            #     print(self.text_feats[a['sentence_id']].shape)


    def __getitem__(self, idx):

        if self.split == 'train':
            data = self.data_list[idx]
            sentence_id = data['sentence_id']
            sentence = data['sentence']
            video_id = data['id']
            visual_feat, overlaps, visual_mask = self._get_video_features_train(video_id, data)
        else:
            window_offset, real_data_index = self._find_windows_num(idx)
            data = self.data_list[real_data_index]
            sentence_id = data['sentence_id']
            sentence = data['sentence']
            video_id = data['id']
            visual_feat, overlaps, visual_mask = self._get_video_features_test(video_id, data, window_offset)

        word_vector = self._get_language_feature(sentence_id, sentence)

        item = {
            'visual_input': visual_feat,
            'vis_mask': visual_mask,
            'anno_idx': idx,
            'word_vectors': word_vector,
            'duration': self.num_pre_clips / 5,
            'txt_mask': torch.ones(word_vector.shape[0], 1),
            'map_gt': overlaps,
        }
        return item

    def __len__(self):
        if self.split == 'test':
            return self.dataset_size

        return len(self.data_list)

    def _get_language_feature(self, sentence_id, sentence):
        '''
            INPUTS:
            sentence_id :  id of the sentence
            sentence : all tokens in the sentnce


            OUTPUTS:
            text_feature: clip text feature
        '''

        # path = os.path.join(self.text_feat_dir, ("{:06d}".format(int(sentence_id)) + ".npy"))
        #
        # if path in self.cache_texts.keys():
        #     text_feature = self.cache_texts[path]
        # else:
        #     text_feature = np.load(path)
        #     text_feature = torch.from_numpy(text_feature.astype(np.float32).transpose(0, 1))
        #     self.cache_texts.update({path: text_feature})

        text_feature = self.text_feats[sentence_id]

        return text_feature

    def _get_video_features_train(self, video_id, data):
        '''
            INPUTS:
            video_id: id of the video
            data: annotation data, contains all the preprocessed information


            OUTPUTS:
            feat: movie features
            iou2d: target matrix
        '''

        self.input_stride = 1

        movie_feature = self.feats[video_id]

        start_sec, stop_sec = data['segment']
        start_idx = math.ceil(start_sec * data['fps'])
        stop_idx = math.floor(stop_sec * data['fps'])

        num_frames = int(stop_idx - start_idx)
        if num_frames < self.num_pre_clips:
            offset = random.sample(range(0, self.num_pre_clips - num_frames, 1), 1)[0]
        else:
            center = (start_idx + stop_idx) / 2
            offset = int(round(center) / 2)  # keep the setting as orig github repo

        # Compute features for window
        start_window = max(int(start_idx) - offset, 0)
        stop_window = start_window + self.num_pre_clips * self.input_stride
        # print(start_window,stop_window)
        if not stop_window < data['num_frames']:
            stop_window = int(data['num_frames'])
            start_window = stop_window - self.num_pre_clips * self.input_stride

        feats = movie_feature[start_window:stop_window, :]

        assert feats.shape[0] == self.num_pre_clips

        # Compute moment position withint the windo
        duration = self.num_pre_clips / data['fps']
        start_moment = max(start_window / data['fps'], 0)
        stop_moment = min(stop_window / data['fps'], duration)

        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0, num_clips).float() * duration / num_clips + start_moment
        e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips + start_moment
        overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                    e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                       torch.tensor([start_sec, stop_sec]).tolist()).reshape(num_clips, num_clips)

        feats = F.normalize(feats, dim=1)

        return feats, torch.from_numpy(overlaps), torch.ones((self.num_pre_clips, 1))

    def _find_windows_num(self, idx):
        '''
            INPUTS:
            idx: idx pass into dataloader


            OUTPUTS:
            window_offset： window offset for current data point
            data_list_index : index for this data in self.anno_list
        '''
        window_offset = 0
        data_list_index = 0
        for key in self.query_interval_index.keys():
            window = self.query_interval_index[key]
            if window[0] <= idx < window[1]:
                # print(window,idx)
                window_offset = idx - window[0]
                data_list_index = window[2]
                break
        return window_offset, data_list_index

    def _get_video_features_test(self, video_id, data, windows_offset):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix
        '''

        self.input_stride = 1
        if video_id in self.cache_videos.keys():
            move_feature = self.cache_videos[video_id]
            keys = list(self.cache_videos.keys())
            if len(keys) >= 3:
                for key in keys:
                    self.cache_videos.pop(key)
        else:

            video_name = video_id + ".npy"
            move_feature = np.load(os.path.join(self.vid_feat_dir, video_name))
            move_feature = torch.from_numpy(move_feature)
            self.cache_videos.update({video_id: move_feature})

        if windows_offset == data['num_windows']:
            feat_end = move_feature.shape[0]
            feat_start = feat_end - self.num_pre_clips
        else:
            feat_start = int(self.test_stride * windows_offset)
            feat_end = int(feat_start + self.num_pre_clips)

        window_feat = move_feature[feat_start:feat_end, :]

        feat = F.normalize(window_feat, dim=1)
        vis_mask = torch.ones((self.num_pre_clips, 1))

        iou = torch.zeros(self.num_clips, self.num_clips)

        return feat, iou, vis_mask

    def check(self, sentence_id):
        path = os.path.join(self.text_feat_dir, ("{:06d}".format(int(sentence_id)) + ".npy"))
        if not os.path.exists(path):
            return sentence_id
        else:
            return None

    def _load_annotation(self):
        self.query_interval_index = {}

        self.anno_file = self.anno_file_path[self.split]

        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        data_list = tuple()
        if self.split == 'train':
            for key, value in tqdm(anno.items()):
                fps = 5
                movie_duration = value["movie_duration"]
                video_id = value['movie']
                num_frames = movie_duration * fps

                start = max(value['ext_timestamps'][0], 0)
                end = min(value['ext_timestamps'][1], movie_duration)
                if start >= end:
                    continue
                sentence = value['sentence']
                data_list += (
                    {'id': video_id,
                     'fps': fps,
                     'num_frames': num_frames,
                     'duration': self.num_pre_clips / fps,
                     'sentence': sentence,
                     'segment': (start, end),
                     "sentence_id": key,
                     'movie_duration':  movie_duration
                     },
                )
        else:
            return data_list
            data_index = 0
            data_list_index = 0
            for key, value in tqdm(anno_db.items()):

                fps, num_frames = value['fps'], value['num_frames']
                self.input_stride = 1

                duration = num_frames / fps

                if 'annotations' in value:
                    for pair in value['annotations']:
                        start = max(pair['segment'][0], 0)
                        end = min(pair['segment'][1], duration)

                        sentence = pair['sentence'].strip().lower()
                        sentence_id = pair['sentence_id']

                        self.window = self.num_pre_clips
                        stride = self.num_pre_clips / 2

                        num_windows = int((num_frames - self.window + stride) // stride)

                        if (num_frames - self.window + stride) % stride != 0:
                            num_windows += 1

                        if int(num_frames) - self.window + stride <= stride:
                            print('warning:', int(num_frames), self.window, stride)

                        self.query_interval_index.update(
                            {sentence_id: (data_index, data_index + num_windows, data_list_index)})

                        data_index += num_windows
                        data_list += (
                            {
                                'id': key,
                                'fps': fps,
                                "duration": config.DATASET.NUM_SAMPLE_CLIPS / fps,
                                "sentence": sentence,
                                "sentence_id": sentence_id,
                                "segment": (start, end),
                                "num_windows": num_windows - 1,
                                "num_frames": num_frames
                            },
                        )
                        data_list_index += 1
            self.dataset_size = data_index
        return data_list

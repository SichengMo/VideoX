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


class MADClipdataset(torch.utils.data.Dataset):

    # text centric dataset
    def __init__(self, split):
        data_dir = os.path.join('data', 'MAD')

        self.sub_test_idx = config.TEST.SUB_TEST_INDEX
        self.chunk_size = config.TEST.CHUCK_SIZE

        # get the annotation file name
        anno_file_name = os.path.join(data_dir, ('train_chuck_{:04d}'.format(self.chunk_size)),
                                      ('mad_test_{:02d}.json'.format(self.sub_test_idx)))
        config.TEST.OUTPATH = os.path.join(data_dir, ('train_chuck_{:04d}'.format(self.chunk_size)),
                                           'out')
        os.makedirs(config.TEST.OUTPATH, exist_ok=True)

        config.TEST.OUTPATH = os.path.join(data_dir, ('train_chuck_{:04d}'.format(self.chunk_size)),
                                           'out', ('mad_test_{:02d}.npy'.format(self.sub_test_idx)))
        # print(self.sub_test_idx)

        #########################
        # cache video araa

        self.cache_videos = {}
        self.cache_texts = {}

        ##########################

        self.split = split

        self.vid_feat_dir = os.path.join(data_dir, '5fps_clip1_s1_512d')
        self.text_feat_dir = os.path.join(data_dir, 'token_512d')
        if self.split == 'test' and self.sub_test_idx != -1:
            self.anno_file = anno_file_name
        else:
            self.anno_file = os.path.join(data_dir, 'mad.json')

        print()

        self.num_pre_clips = config.DATASET.NUM_SAMPLE_CLIPS
        self.test_stride = self.num_pre_clips / 2
        self.target_stride = config.DATASET.TARGET_STRIDE

        self.num_clips = int(self.num_pre_clips / self.target_stride)

        self.videos = glob(os.path.join(self.vid_feat_dir, ('*.npy')))
        self.videos = [video for video in self.videos]

        cache_file = os.path.join(data_dir, (split + '.json'))
        self.missing = []

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

        for each in sorted(self.missing):
            print(each)

        missing = {'missing': sorted(self.missing)}
        with open(os.path.join(data_dir, ('mad_' + self.split + '_missing' + '.json')), 'w') as f:
            json.dump(missing, f)

        # print(self.videos)
        print(self.__len__())
        self.annotations = self.data_list

    def __getitem__(self, idx):

        data = self.data_list[idx]
        sentence_id = data['sentence_id']
        sentence = data['sentence']
        duration = data['duration']
        video_id = data['id']

        if self.split == 'train':
            feat, iou2d, visual_mask = self._get_video_features_train(video_id, data)
        else:
            feat, iou2d, visual_mask = self._get_video_features_test(video_id, data)

        word_vectors = self._get_language_feature(sentence_id, sentence)
        index = idx

        visual_input = feat
        overlaps = iou2d
        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': self.num_pre_clips / 5,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': overlaps,
        }
        # print(visual_input.shape)
        return item

    def __len__(self):
        return len(self.data_list)

    def _get_language_feature(self, sentence_id, sentence):
        '''
            INPUTS:
            sentence_id :  id of the sentence
            sentence : all tokens in the sentnce


            OUTPUTS:
            text_feature: clip text feature
        '''

        path = os.path.join(self.text_feat_dir, ("{:06d}".format(int(sentence_id)) + ".npy"))

        if path in self.cache_texts.keys():
            text_feature = self.cache_texts[path]
        else:
            text_feature = np.load(path)
            text_feature = torch.from_numpy(text_feature.astype(np.float32).transpose(0, 1))
            self.cache_texts.update({path: text_feature})

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
        video_name = ''

        # first load the video
        for name in self.videos:
            if video_id in name:
                video_name = name
                break

        movie_feature = np.load(video_name)
        movie_feature = torch.from_numpy(movie_feature.astype(np.float32))

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

    def _get_video_features_test(self, video_id, data):
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
            moive_feature = self.cache_videos[video_id]
            keys = list(self.cache_videos.keys())
            if len(keys) >= 5:
                for key in keys:
                    self.cache_videos.pop(key)
        else:

            video_name = video_id + ".npy"
            moive_feature = np.load(os.path.join(self.vid_feat_dir, video_name))
            moive_feature = torch.from_numpy(moive_feature)
            self.cache_videos.update({video_id: moive_feature})

        window_from_orig = data['window_from_orig']
        moive_feature = moive_feature[window_from_orig[0]:window_from_orig[1], :]
        # window_se = data['window']

        w_start = data['window'][0]
        w_end = data['window'][1]

        window_feat = moive_feature[w_start:w_end, :]

        feat = F.normalize(window_feat, dim=1)
        vis_mask = torch.ones((self.num_pre_clips, 1))

        iou = torch.zeros(self.num_clips, self.num_clips)

        return feat, iou, vis_mask

    def _load_annotation(self):
        self.missing = []
        self.query_interval_index = {}
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        # combine data from all splits
        anno_db = dict()
        for s in [self.split]:
            assert s in anno, 'split does not exist'
            anno_db.update(anno[s])

        data_list = tuple()
        if self.split == 'train':
            # do nothing here
            _ = 0
        else:
            for key, value in tqdm(anno_db.items()):
                fps, num_frames = value['fps'], value['num_frames']
                self.input_stride = 1

                video_id = key.split('_???_')[0]
                duration = num_frames / fps

                if 'annotations' in value:
                    window_from_orig = value['window']

                    for pair in value['annotations']:

                        start = max(pair['segment'][0], 0)
                        end = min(pair['segment'][1], duration)
                        sentence = pair['sentence'].strip().lower()
                        id = pair['sentence_id']

                        clip_duration = num_frames

                        self.window = self.num_pre_clips
                        stride = self.num_pre_clips / 2
                        num_windows = int((num_frames - self.window + stride) // stride)

                        if (num_frames - self.window + stride) % stride != 0:
                            num_windows += 1

                        if int(clip_duration) - self.window + stride <= stride:
                            print('warning:', int(clip_duration), self.window, stride)

                        for i in range(num_windows):
                            w_start = stride * i
                            w_end = w_start + self.window
                            if w_end >= num_frames:
                                w_end = num_frames
                                w_start = num_frames - self.window
                                w_end = int(w_end)
                                w_start = int(w_start)
                                data_list += (
                                    {
                                        'id': video_id,
                                        'fps': fps,
                                        "duration": config.DATASET.NUM_SAMPLE_CLIPS / fps,
                                        "sentence": sentence,
                                        "window": [w_start, w_end],
                                        "sentence_id": id,
                                        "segment": (start, end),
                                        'last_window': True,
                                        "num_windows": num_windows - 1,
                                        "num_frames": num_frames,
                                        "window_from_orig": (window_from_orig[0], window_from_orig[1])
                                    },
                                )
                            else:
                                w_end = int(w_end)
                                w_start = int(w_start)
                                data_list += (
                                    {
                                        'id': video_id,
                                        'fps': fps,
                                        "duration": config.DATASET.NUM_SAMPLE_CLIPS / fps,
                                        "sentence": sentence,
                                        "window": [w_start, w_end],
                                        "sentence_id": id,
                                        "segment": (start, end),
                                        'last_window': False,
                                        "num_windows": num_windows - 1,
                                        "num_frames": num_frames,
                                        "window_from_orig": (window_from_orig[0], window_from_orig[1])
                                    },
                                )
        return data_list

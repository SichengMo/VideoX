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

from transformers import CLIPTokenizer, CLIPModel


class MADTestdataset(torch.utils.data.Dataset):

    # text centric dataset
    def __init__(self, split):

        #########################
        # cache video araa

        self.cache_videos = {}


        ##########################



        self.split = split
        print(split)
        data_dir = os.path.join('data', 'MAD')
        self.vid_feat_dir = os.path.join(data_dir, '5fps_clip1_s1_512d')
        self.text_feat_dir = os.path.join(data_dir, 'token_512d')
        self.anno_file = os.path.join(data_dir, 'mad.json')

        self.num_pre_clips = 128
        self.target_stride = 8
        self.num_clips = int(self.num_pre_clips / self.target_stride)

        # self.processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.videos = glob(os.path.join(self.vid_feat_dir, ('*.npy')))
        self.videos = [video for video in self.videos]
        print(self.num_clips)

        cache_file = os.path.join(data_dir, (split + '.json'))
        self.missing =[]

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
        with open(os.path.join(data_dir, ('mad_'+self.split+'_missing'+'.json')),'w') as f:
            json.dump(missing,f)


        # print(self.videos)
        print(self.__len__())
        self.annotations = self.data_list
    def __getitem__(self, idx):

        # if not self.split == 'test':
        data = self.data_list[idx]
        # else:
        #     window_offset,real_data_index = self._find_windows_num(idx)
        #     data = self.data_list[real_data_index]

        sentence_id = data['sentence_id']
        sentence = data['sentence']
        duration = data['duration']
        video_id = data['id']

        if self.split == 'train':
            feat, iou2d, visual_mask = self._get_video_features_train(video_id, data)
        else:
            # not working now
            feat, iou2d,visual_mask = self._get_video_features_test(video_id, data,0)

        #feat = average_to_fixed_length(feat)

        word_vectors = self._get_language_feature(sentence_id, sentence)
        index = idx

        visual_input = feat
        overlaps = iou2d
        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': overlaps,
        }
        return item

    def __len__(self):

        return len(self.data_list)

    def _get_language_feature(self, sentence_id, sentence):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information

            OUTPUTS:
            query: features of the selected sentence
            wordlen: length of the selected sentence
        '''
        try:
            path = os.path.join(self.text_feat_dir, ("{:06d}".format(int(sentence_id)) + ".npy"))
            text_feature = np.load(path)
            text_feature = torch.from_numpy(text_feature.astype(np.float32).transpose(0, 1))
        except:
            inputs = self.processor(sentence)
            text_feature = self.clip_model(**inputs).last_hidden_state
            print(text_feature.shape)
        return text_feature

    def _get_video_features_train(self, video_id, data):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

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

        moive_feature = np.load(video_name)
        moive_feature = torch.from_numpy(moive_feature)



        start_idx, stop_idx = data['segment']
        start_idx = start_idx* data['fps']
        stop_idx = stop_idx * data['fps']
        num_frames = int(stop_idx - start_idx)
        if num_frames < self.num_pre_clips:
            offset = random.sample(range(0, self.num_pre_clips - num_frames, 1), 1)[0]
        else:
            center = (start_idx + stop_idx) / 2
            offset = int(round(center / 2))

        # Compute features for window
        start_window = max(int(start_idx) - offset, 0)
        stop_window = start_window + self.num_pre_clips * self.input_stride
        # print(start_window,stop_window)
        if not stop_window < data['num_frames']:
            stop_window = int(data['num_frames'])
            start_window = stop_window - self.num_pre_clips * self.input_stride

        feats = moive_feature[start_window:stop_window,:]

        if feats.shape[0] != self.num_pre_clips:
            print(start_idx,stop_idx)
            print(data['num_frames'])
            print(moive_feature.shape)
            print(feats.shape)
            print(feats.shape[0])
            print(start_window)
            print(stop_window)
            print(data['duration'] * data['fps'])
            print(self.num_pre_clips)
            print("____________________________________________-")
        assert feats.shape[0] == self.num_pre_clips

        # Compute moment position withint the windo
        duration = self.num_pre_clips / data['fps']
        start_moment = max((start_idx - start_window) / data['fps'], 0)
        stop_moment = min((stop_idx - start_window) / data['fps'], duration)


        num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0,num_clips).float()*duration/num_clips
        e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
        overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                    e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)














        moment = torch.tensor([start_moment, stop_moment])
        # Generate targets for training ----------------------------------------------
        iou2d = moment_to_iou2d(moment, self.num_clips, duration)
        # print(iou2d.shape)

        feats = F.normalize(feats, dim=1)

        return feats, iou2d, torch.ones((128, 1))

    def _find_windows_num(self,idx):
        window_offset = 0
        data_list_inde = 0
        for key in self.query_interval_index.keys():
            window = self.query_interval_index[key]
            if window[0] <= idx < window[1]:
                # print(window,idx)
                window_offset = idx - window[0]
                data_list_index = window[2]
                break
        return window_offset,data_list_index



    def _get_window_data(self,video_feature,window):
        return

    def _get_video_features_test(self, video_id,data,windows_offset):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix
        '''

        self.input_stride = 1

        # search if it is in cache videos
        if video_id in self.cache_videos.keys():
            movie_feature = self.cache_videos[video_id]
        else:
            # load from video file
            video_name = ''

            # first load the video
            for name in self.videos:
                if video_id in name:
                    video_name = name
                    break

            moive_feature = np.load(video_name)
            moive_feature = torch.from_numpy(moive_feature)

        # feat_start = int(64*windows_offset)
        # feat_end = int(feat_start+128)
        # # print(feat_start)
        # # print(feat_end)
        # # print(moive_feature.shape)
        # # print("________________________________________")
        # try:
        #     window_feat = moive_feature[feat_start:feat_end,:]
        # except:
        #     window_feat = moive_feature[feat_start:,:]
        # feat = window_feat
        feat = moive_feature

        feat = F.normalize(feat,dim=1)
        vis_mask = torch.ones((128,1))

        iou = torch.zeros(16,16)

        return feat,iou,vis_mask

    def check(self,sentence_id):
        path = os.path.join(self.text_feat_dir, ("{:06d}".format(int(sentence_id)) + ".npy"))
        if not os.path.exists(path):
            return sentence_id
        else:
            return None

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
        for key, value in tqdm(anno_db.items()):
            fps, num_frames = value['fps'], value['num_frames']
            duration = num_frames / fps
            if 'annotations' in value:
                for pair in value['annotations']:
                    start = max(pair['segment'][0], 0)
                    end = min(pair['segment'][1], duration)
                    if start >= end:
                        continue
                    sentence = pair['sentence'].strip().lower()
                    id = pair['sentence_id']
                    missing = self.check(id)
                    if missing != None:
                        self.missing.append(missing)
                        continue
                    data_list += (
                        {'id': key,
                         'fps': fps,
                         'num_frames': num_frames,
                         'duration': duration,
                         'sentence': sentence,
                         'segment': (start, end),
                         "sentence_id": id
                         },
                    )
        return data_list

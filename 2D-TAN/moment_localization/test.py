import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_predictions, display_results,nms,eval
import models.loss as loss

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose

def save_scores(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d['video']] = scores[i]
    pkl.dump(results,open(os.path.join(config.RESULT_DIR, dataset_name, '{}_{}_{}.pkl'.format(config.MODEL.NAME,config.DATASET.VIS_INPUT_TYPE,
        split)),'wb'))

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = getattr(models, config.MODEL.NAME)()
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(device)
    model.eval()

    global test_dataset
    test_dataset = getattr(datasets, config.DATASET.NAME)(args.split)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.collate_fn)

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()

        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        prediction, map_mask = model(textual_input, textual_mask, visual_input)
        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times

    # def get_proposal_results(scores, durations):
    #     # assume all valid scores are larger than one
    #     out_sorted_times = []
    #     for score, duration in zip(scores, durations):
    #         T = score.shape[-1]
    #         sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
    #         sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
    #
    #         sorted_indexs[:,1] = sorted_indexs[:,1] + 1
    #         sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
    #         target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
    #         out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())
    #
    #     return out_sorted_times

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            score_cpu = score.cpu().detach().numpy()
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score_cpu.ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
            sorted_scores = np.array([score_cpu[0, int(x[0]),int(x[1])] for x in sorted_indexs])

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1

            # print(sorted_indexs[0])
            # print(sorted_indexs[1])
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()

            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            sorted_time = (sorted_indexs.float() / target_size * duration).tolist()
            # print(sorted_time[0])
            # print(sorted_time[1])
            # exit()
            out_sorted_times.append([[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)])

        return out_sorted_times

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['output'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TRAIN.BATCH_SIZE))

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)


    def post_process(segments, data, verbose=True, merge_window=False, run_eval=True):
        if merge_window:
            merge_seg = {}
            merge_data = {}
            idxs = list(np.arange(0,len(segments),1))
            for seg, idx in zip(segments, idxs):
                window_offset,real_data_index = test_dataset._find_windows_num(idx)
                dat = data[real_data_index]
                pair_id = dat['sentence_id']  # dat['anno_uid'] + '_' + str(dat['query_idx'])
                if pair_id not in merge_seg.keys():  # new
                    merge_data[pair_id] = {
                        'video': dat['id'],
                        'duration':128/5,
                        'segment': [dat['segment'][0],dat['segment'][1]],
                        'sentence': dat['sentence'],
                        #'query_idx': dat['query_idx'],
                    }
                    merge_seg[pair_id] = []
                #print(idx,window_offset)

                offset = window_offset*64/5
                # offset = dat['window'][0]
                #print(seg)
                merge_seg[pair_id].extend([[se[0] + offset, se[1] + offset, se[2]] for se in seg])
            # print(merge_seg.keys())
            segments, data = [], []
            for k in merge_seg.keys():
                segments.append(sorted(merge_seg[k], key=lambda x: x[2], reverse=True))
                data.append(merge_data[k])

        segments = [nms(seg, thresh=config.TEST.NMS_THRESH, top_k=50).tolist() for seg in segments]

        if run_eval:
            eval_result, miou = eval(segments, data)
        else:
        #     save_prediction_to_file(config.RESULT, merge_seg, merge_data)
        #     print('Prediction result is saved to {}'.format(config.RESULT))
             eval_result, miou = 0, 0

        return eval_result, miou

    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = test_dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = post_process(state['sorted_segments_list'], annotations, True,merge_window=True,run_eval=True)

        loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
        print(loss_message)
        state['loss_meter'].reset()
        test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                          'performance on testing set')
        table_message = '\n'+test_table
        print(table_message)

        # save_scores(state['sorted_segments_list'], annotations, config.DATASET.NAME, args.split)


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network,dataloader, args.split)

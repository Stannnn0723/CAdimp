import numpy as np
import multiprocessing
import os
import sys
import csv
import json
import cv2
from itertools import product
from collections import OrderedDict
from pytracking.evaluation import Sequence, Tracker
from ltr.data.image_loader import imwrite_indexed


PREDICTION_FIELD_NAMES = ['video', 'object', 'frame_num', 'present', 'score', 'xmin', 'xmax', 'ymin', 'ymax']


def _get_diag_config(tracker: Tracker):
    """Read diagnostic config from tracker parameters with safe defaults."""
    try:
        params = tracker.get_parameters()
    except Exception:
        return {
            'enabled': False,
            'failure_iou_threshold': 0.30,
            'dump_visual': True,
            'dump_max_frames': 60,
        }

    return {
        'enabled': bool(params.get('diag_debug_mode', False)),
        'failure_iou_threshold': float(params.get('diag_failure_iou_thresh', 0.30)),
        'dump_visual': bool(params.get('diag_dump_visual_overlay', True)),
        'dump_max_frames': int(params.get('diag_dump_max_frames', 60)),
    }


def _bbox_iou_xywh(box_a, box_b):
    """Compute IoU for two xywh boxes."""
    if box_a is None or box_b is None:
        return None

    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    ix1 = max(a_x1, b_x1)
    iy1 = max(a_y1, b_y1)
    ix2 = min(a_x2, b_x2)
    iy2 = min(a_y2, b_y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _compute_sequence_metrics(seq: Sequence, output: dict):
    """Compute per-frame IoU, mean IoU and SR@0.5 for single-object sequences."""
    gt = seq.ground_truth_rect
    pred = output.get('target_bbox', None)
    if gt is None or pred is None:
        return None
    if isinstance(gt, (dict, OrderedDict)):
        return None
    if len(pred) == 0 or isinstance(pred[0], (dict, OrderedDict)):
        return None

    n = min(len(gt), len(pred))
    ious = []
    for i in range(n):
        p = pred[i]
        g = gt[i]
        if p is None or len(p) != 4:
            ious.append(0.0)
            continue
        if p[0] < 0 or p[1] < 0 or p[2] <= 0 or p[3] <= 0:
            ious.append(0.0)
            continue
        iou = _bbox_iou_xywh(p, g)
        ious.append(0.0 if iou is None else float(iou))

    if len(ious) == 0:
        return None

    mean_iou = float(np.mean(ious))
    sr50 = float(np.mean(np.array(ious) >= 0.5))
    return {'ious': ious, 'mean_iou': mean_iou, 'sr50': sr50}


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return bool(v)


def _dump_failure_visuals(seq: Sequence, tracker: Tracker, output: dict, ious, dump_dir, max_frames):
    """Dump overlay panels for hardest frames in a failed sequence."""
    frames_dir = os.path.join(dump_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    pred = output.get('target_bbox', [])
    diag = output.get('diagnostic_state', [])
    score_maps = output.get('diag_score_map', [])

    n = min(len(seq.frames), len(pred), len(ious))
    if n <= 1:
        return

    candidate_idx = list(range(1, n))
    candidate_idx.sort(key=lambda idx: ious[idx])
    selected = candidate_idx[:max_frames]

    for idx in selected:
        frame = cv2.imread(seq.frames[idx])
        if frame is None:
            continue

        panel_left = frame.copy()
        panel_right = frame.copy()

        pred_box = pred[idx]
        if pred_box is not None and len(pred_box) == 4 and pred_box[2] > 0 and pred_box[3] > 0:
            x, y, w, h = [int(v) for v in pred_box]
            cv2.rectangle(panel_left, (x, y), (x + w, y + h), (0, 255, 0), 2)

        gt_box = None
        if seq.ground_truth_rect is not None and not isinstance(seq.ground_truth_rect, (dict, OrderedDict)) and idx < len(seq.ground_truth_rect):
            gt_box = seq.ground_truth_rect[idx]
        if gt_box is not None and len(gt_box) == 4 and gt_box[2] > 0 and gt_box[3] > 0:
            gx, gy, gw, gh = [int(v) for v in gt_box]
            cv2.rectangle(panel_left, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)

        score_map = score_maps[idx] if idx < len(score_maps) else None
        if score_map is not None:
            score_map = np.asarray(score_map, dtype=np.float32)
            s_min = float(score_map.min())
            s_max = float(score_map.max())
            if s_max - s_min > 1e-8:
                score_map = (score_map - s_min) / (s_max - s_min)
            else:
                score_map = np.zeros_like(score_map)
            score_uint8 = (score_map * 255.0).clip(0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (panel_right.shape[1], panel_right.shape[0]), interpolation=cv2.INTER_LINEAR)
            panel_right = cv2.addWeighted(panel_right, 0.45, heat, 0.55, 0.0)

        diag_state = diag[idx] if idx < len(diag) and isinstance(diag[idx], dict) else {}
        score_txt = _safe_float(diag_state.get('max_score', None))
        hscale_txt = _safe_float(diag_state.get('H_p95_mean', None))
        attack_txt = _safe_bool(diag_state.get('is_attacked', False))
        anchor_txt = _safe_bool(diag_state.get('anchor_updated', False))
        text = 'Score: {:.3f} | Attack: {} | Anchor Update: {} | H_Scale: {}'.format(
            score_txt if score_txt is not None else -1.0,
            str(attack_txt),
            str(anchor_txt),
            '{:.4f}'.format(hscale_txt) if hscale_txt is not None else 'None',
        )

        panel = np.concatenate([panel_left, panel_right], axis=1)
        cv2.putText(panel, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, 'IoU: {:.3f} | Frame: {}'.format(ious[idx], idx), (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        out_name = 'frame_{:04d}.jpg'.format(idx)
        cv2.imwrite(os.path.join(frames_dir, out_name), panel)


def _dump_failure_case(seq: Sequence, tracker: Tracker, output: dict, metrics: dict, cfg: dict):
    """Dump per-frame diagnostic CSV/JSON and optional overlay images for failed sequences."""
    dump_dir = os.path.join(tracker.results_dir, 'diagnostic_failures', seq.name)
    os.makedirs(dump_dir, exist_ok=True)

    ious = metrics['ious']
    diag = output.get('diagnostic_state', [])
    pred = output.get('target_bbox', [])
    n = min(len(seq.frames), len(pred), len(ious))

    csv_path = os.path.join(dump_dir, 'diag_timeline.csv')
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            'frame_idx', 'frame_path', 'iou', 'max_score', 'dist_from_center',
            'is_attacked', 'is_lost', 'anchor_updated', 'H_p95_mean', 'pred_bbox'
        ])
        writer.writeheader()
        for i in range(n):
            ds = diag[i] if i < len(diag) and isinstance(diag[i], dict) else {}
            writer.writerow({
                'frame_idx': i,
                'frame_path': seq.frames[i],
                'iou': ious[i],
                'max_score': ds.get('max_score', None),
                'dist_from_center': ds.get('dist_from_center', None),
                'is_attacked': ds.get('is_attacked', False),
                'is_lost': ds.get('is_lost', False),
                'anchor_updated': ds.get('anchor_updated', False),
                'H_p95_mean': ds.get('H_p95_mean', None),
                'pred_bbox': pred[i],
            })

    summary = {
        'sequence': seq.name,
        'dataset': seq.dataset,
        'num_frames': n,
        'mean_iou': metrics['mean_iou'],
        'sr50': metrics['sr50'],
        'failure_iou_threshold': cfg['failure_iou_threshold'],
        'csv': csv_path,
    }
    json_path = os.path.join(dump_dir, 'summary.json')
    with open(json_path, 'w') as fp:
        json.dump(summary, fp, indent=2)

    if cfg.get('dump_visual', True):
        _dump_failure_visuals(seq, tracker, output, ious, dump_dir, cfg.get('dump_max_frames', 60))


def _normalize_gpu_ids(gpu_ids):
    """Normalize gpu_ids to a list of ints, or empty list when disabled."""
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_ids.strip()
        if gpu_ids == '':
            return []
        return [int(x.strip()) for x in gpu_ids.split(',') if x.strip() != '']
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    return [int(x) for x in gpu_ids]


def _run_sequence_with_gpu(seq, tracker, debug=False, visdom_info=None, gpu_id=None):
    """Run one sequence while binding this worker process to a specific GPU."""
    if gpu_id is not None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(int(gpu_id))
        except Exception:
            pass

    return run_sequence(seq, tracker, debug=debug, visdom_info=visdom_info)


def _run_sequence_with_gpu_star(args):
    """Tuple-arg wrapper for imap_unordered in multiprocessing pool."""
    return _run_sequence_with_gpu(*args)


def _save_tracker_output_oxuva(seq: Sequence, tracker: Tracker, output: dict):
    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    img_h, img_w = output['image_shape']
    tracked_bb = np.array(output['target_bbox'])
    object_presence_scores = np.array(output['object_presence_score'])

    tracked_bb = np.vstack([
        tracked_bb[:, 0]/img_w,
        (tracked_bb[:, 0] + tracked_bb[:, 2])/img_w,
        tracked_bb[:, 1]/img_h,
        (tracked_bb[:, 1] + tracked_bb[:, 3])/img_h,
    ]).T
    tracked_bb = tracked_bb.clip(0., 1.)

    tracked_bb = tracked_bb[1:]
    object_presence_scores = object_presence_scores[1:]
    frame_numbers = np.array(list(map(int, frame_names[1:])))
    vid_id, obj_id = seq.name.split('_')[:2]

    pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))

    with open(pred_file, 'w') as fp:
        writer = csv.DictWriter(fp, fieldnames=PREDICTION_FIELD_NAMES)

        for i in range(0, len(frame_numbers)):
            row = {
                'video': vid_id,
                'object': obj_id,
                'frame_num': frame_numbers[i],
                'present': str(object_presence_scores[i] > output['object_presence_score_threshold']).lower(),  # True or False
                'score': object_presence_scores[i],
                'xmin': tracked_bb[i, 0],
                'xmax': tracked_bb[i, 1],
                'ymin': tracked_bb[i, 2],
                'ymax': tracked_bb[i, 3],
            }
            writer.writerow(row)


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    base_results_path = os.path.join(tracker.results_dir, seq.name)
    segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(float)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%.4f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'segmentation':
            assert len(frame_names) == len(data)
            if not os.path.exists(segmentation_path):
                os.makedirs(segmentation_path)
            for frame_name, frame_seg in zip(frame_names, data):
                imwrite_indexed(os.path.join(segmentation_path, '{}.png'.format(frame_name)), frame_seg)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None):
    """Runs a tracker on a sequence."""

    def _results_exist():
        if seq.dataset == 'oxuva':
            vid_id, obj_id = seq.name.split('_')[:2]
            pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
            return os.path.isfile(pred_file)
        elif seq.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('Skip existing result: {}'.format(seq.name))
        return -1.0

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            return None

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    diag_cfg = _get_diag_config(tracker)
    if diag_cfg.get('enabled', False):
        metrics = _compute_sequence_metrics(seq, output)
        if metrics is not None and metrics['mean_iou'] < diag_cfg['failure_iou_threshold']:
            _dump_failure_case(seq, tracker, output, metrics, diag_cfg)
            print('Diagnostic dump saved for failed sequence: {} | mean IoU {:.3f}'.format(seq.name, metrics['mean_iou']))

    if not debug:
        if seq.dataset == 'oxuva':
            _save_tracker_output_oxuva(seq, tracker, output)
        else:
            _save_tracker_output(seq, tracker, output)

    return num_frames / exec_time


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None, gpu_ids=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
        gpu_ids: Optional GPU id list/string, e.g. [0,1] or '0,1'.
                 In parallel mode, jobs are assigned round-robin across GPUs.
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    visdom_info = {} if visdom_info is None else visdom_info
    gpu_ids = _normalize_gpu_ids(gpu_ids)

    if threads == 0 and len(gpu_ids) > 1:
        threads = len(gpu_ids)

    total_seq_count = len(dataset) * len(trackers)
    completed = 0
    fps_list = []

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        seq_tracker_list = list(product(dataset, trackers))
        for job_idx, (seq, tracker_info) in enumerate(seq_tracker_list):
            gpu_id = gpu_ids[job_idx % len(gpu_ids)] if len(gpu_ids) > 0 else None
            fps = _run_sequence_with_gpu(seq, tracker_info, debug=debug, visdom_info=visdom_info, gpu_id=gpu_id)
            completed += 1
            if fps is not None:
                fps_list.append(fps)
            progress_pct = completed / total_seq_count * 100
            print('\rProgress: {:.1f}% ({}/{})'.format(progress_pct, completed, total_seq_count), end='')
            sys.stdout.flush()
    elif mode == 'parallel':
        seq_tracker_list = list(product(dataset, trackers))
        if len(gpu_ids) > 0:
            param_list = [
                (seq, tracker_info, debug, visdom_info, gpu_ids[job_idx % len(gpu_ids)])
                for job_idx, (seq, tracker_info) in enumerate(seq_tracker_list)
            ]
        else:
            param_list = [(seq, tracker_info, debug, visdom_info, None) for seq, tracker_info in seq_tracker_list]

        with multiprocessing.Pool(processes=threads) as pool:
            for fps in pool.imap_unordered(_run_sequence_with_gpu_star, param_list):
                completed += 1
                if fps is not None:
                    fps_list.append(fps)
                progress_pct = completed / total_seq_count * 100
                print('\rProgress: {:.1f}% ({}/{})'.format(progress_pct, completed, total_seq_count), end='')
                sys.stdout.flush()
        print('')
        print('Progress: 100.0% ({}/{})'.format(total_seq_count, total_seq_count))

    print()
    valid_fps = [f for f in fps_list if f > 0]
    if valid_fps:
        avg_fps = sum(valid_fps) / len(valid_fps)
        print('Average FPS: {:.2f}'.format(avg_fps))

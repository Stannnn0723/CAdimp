import argparse
import ast
import csv
import json
import os
from statistics import median


def _to_float(v):
    if v is None or v == '' or str(v).lower() == 'none':
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ('1', 'true', 't', 'yes', 'y')


def _to_bbox(v):
    if v is None or v == '' or str(v).lower() == 'none':
        return None
    try:
        b = ast.literal_eval(str(v))
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return [float(x) for x in b]
    except Exception:
        return None
    return None


def _read_diag_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append({
                'frame_idx': int(r.get('frame_idx', 0)),
                'iou': _to_float(r.get('iou')),
                'max_score': _to_float(r.get('max_score')),
                'dist_from_center': _to_float(r.get('dist_from_center')),
                'is_attacked': _to_bool(r.get('is_attacked')),
                'is_lost': _to_bool(r.get('is_lost')),
                'anchor_updated': _to_bool(r.get('anchor_updated')),
                'h_scale': _to_float(r.get('H_p95_mean')),
                'pred_bbox': _to_bbox(r.get('pred_bbox')),
            })
    return rows


def _consecutive_true_max(flags):
    best = 0
    cur = 0
    for f in flags:
        if f:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _safe_median(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(median(vals))


def _analyze_one(rows, args):
    n = len(rows)
    if n == 0:
        return {
            'label': 'UNKNOWN',
            'reason': 'empty timeline',
            'A_score': 0,
            'B_score': 0,
        }

    lost_flags = [r['is_lost'] for r in rows]
    attacked_flags = [r['is_attacked'] for r in rows]
    anchor_flags = [r['anchor_updated'] for r in rows]

    lost_ratio = sum(1 for x in lost_flags if x) / n
    attacked_ratio = sum(1 for x in attacked_flags if x) / n
    lost_streak = _consecutive_true_max(lost_flags)

    h_vals = [r['h_scale'] for r in rows]
    base_end = max(1, int(n * args.base_portion))
    tail_start = max(0, int(n * (1.0 - args.tail_portion)))
    h_base = _safe_median(h_vals[:base_end])
    h_tail = _safe_median(h_vals[tail_start:])
    h_drop_ratio = None
    if h_base is not None and h_tail is not None and h_base > 1e-12:
        h_drop_ratio = h_tail / h_base

    attacked_anchor_updates = 0
    attacked_count = 0
    for a, u in zip(attacked_flags, anchor_flags):
        if a:
            attacked_count += 1
            if u:
                attacked_anchor_updates += 1
    anchor_update_when_attacked = attacked_anchor_updates / attacked_count if attacked_count > 0 else 0.0

    high_score_offcenter = 0
    offcenter_valid = 0
    for r in rows:
        s = r['max_score']
        d = r['dist_from_center']
        if s is None or d is None:
            continue
        offcenter_valid += 1
        if s > args.bg_score_th and d > args.bg_radius:
            high_score_offcenter += 1
    offcenter_ratio = high_score_offcenter / offcenter_valid if offcenter_valid > 0 else 0.0

    # Probe-1: score derivative fast-drop detector.
    # If score plunges and quickly transitions to lost state, mark as FAST_MOTION signal.
    score_drop_events = 0
    max_scores = [r['max_score'] for r in rows]
    for i in range(1, n):
        prev_s = max_scores[i - 1]
        cur_s = max_scores[i]
        if prev_s is None or cur_s is None:
            continue
        score_diff = cur_s - prev_s
        if score_diff < args.score_drop_th:
            end_i = min(n, i + 1 + args.fast_fail_window)
            if any(lost_flags[j] for j in range(i, end_i)):
                score_drop_events += 1

    fast_motion_score = 1 if score_drop_events > 0 else 0

    # Probe-2: bbox area-ratio collapse detector.
    # If confidence is still high but bbox area jumps abruptly, regression is unstable.
    bbox_collapse_events = 0
    for i in range(1, n):
        prev_b = rows[i - 1]['pred_bbox']
        cur_b = rows[i]['pred_bbox']
        cur_s = rows[i]['max_score']
        if prev_b is None or cur_b is None or cur_s is None:
            continue
        if cur_s <= args.bbox_probe_score_th:
            continue

        prev_area = max(prev_b[2], 0.0) * max(prev_b[3], 0.0)
        cur_area = max(cur_b[2], 0.0) * max(cur_b[3], 0.0)
        if prev_area <= 1e-12:
            continue

        area_ratio = cur_area / prev_area
        if area_ratio > args.bbox_area_ratio_high or area_ratio < args.bbox_area_ratio_low:
            bbox_collapse_events += 1

    bbox_collapse_score = 1 if bbox_collapse_events > 0 else 0

    # Cause A: long occlusion + scale-energy collapse
    A_score = 0
    A_signals = []
    if h_drop_ratio is not None and h_drop_ratio < args.h_drop_ratio_th:
        A_score += 1
        A_signals.append('h_scale collapse')
    if lost_ratio > args.lost_ratio_th:
        A_score += 1
        A_signals.append('high lost ratio')
    if lost_streak >= args.lost_streak_th:
        A_score += 1
        A_signals.append('long lost streak')

    # Cause B: distractor hijack + anchor drift during attack
    B_score = 0
    B_signals = []
    if attacked_ratio > args.attacked_ratio_th:
        B_score += 1
        B_signals.append('high attacked ratio')
    if offcenter_ratio > args.offcenter_ratio_th:
        B_score += 1
        B_signals.append('high-score off-center peaks')
    if anchor_update_when_attacked > args.anchor_attack_update_th:
        B_score += 1
        B_signals.append('anchor updates during attack')

    if A_score >= 2 and B_score < 2:
        label = 'A_SCALE_COLLAPSE'
        reason = '; '.join(A_signals)
    elif B_score >= 2 and A_score < 2:
        label = 'B_ANCHOR_POISONING'
        reason = '; '.join(B_signals)
    elif A_score >= 2 and B_score >= 2:
        label = 'MIXED'
        reason = 'A: {} | B: {}'.format(', '.join(A_signals), ', '.join(B_signals))
    elif B_score == 1 and A_score == 0:
        label = 'WEAK_ANCHOR_POISONING'
        reason = 'early distractor-hijack symptom ({})'.format(', '.join(B_signals))
    elif fast_motion_score > 0 and bbox_collapse_score > 0:
        label = 'FAST_MOTION_BBOX_COLLAPSE'
        reason = 'fast score plunge + high-confidence bbox area jump'
    elif fast_motion_score > 0:
        label = 'FAST_MOTION'
        reason = 'score_diff drops below threshold and quickly transitions to lost'
    elif bbox_collapse_score > 0:
        label = 'BBOX_COLLAPSE'
        reason = 'high-confidence bbox area ratio outlier'
    else:
        label = 'UNKNOWN'
        reason = 'insufficient strong signals'

    return {
        'label': label,
        'reason': reason,
        'A_score': A_score,
        'B_score': B_score,
        'num_frames': n,
        'lost_ratio': lost_ratio,
        'lost_streak': lost_streak,
        'attacked_ratio': attacked_ratio,
        'anchor_update_when_attacked': anchor_update_when_attacked,
        'offcenter_ratio': offcenter_ratio,
        'fast_motion_score': fast_motion_score,
        'score_drop_events': score_drop_events,
        'bbox_collapse_score': bbox_collapse_score,
        'bbox_collapse_events': bbox_collapse_events,
        'h_base': h_base,
        'h_tail': h_tail,
        'h_drop_ratio': h_drop_ratio,
    }


def _find_sequence_dirs(root):
    seq_dirs = []
    if not os.path.isdir(root):
        return seq_dirs
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        csv_path = os.path.join(path, 'diag_timeline.csv')
        if os.path.isfile(csv_path):
            seq_dirs.append(path)
    return seq_dirs


def main():
    parser = argparse.ArgumentParser(description='Auto label failure causes from diagnostic timeline CSVs.')
    parser.add_argument('--diag_root', type=str, required=True,
                        help='Path to diagnostic_failures root directory.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path. Defaults to <diag_root>/auto_cause_summary.json')

    parser.add_argument('--base_portion', type=float, default=0.2)
    parser.add_argument('--tail_portion', type=float, default=0.3)
    parser.add_argument('--h_drop_ratio_th', type=float, default=0.35)
    parser.add_argument('--lost_ratio_th', type=float, default=0.25)
    parser.add_argument('--lost_streak_th', type=int, default=15)
    parser.add_argument('--attacked_ratio_th', type=float, default=0.20)
    parser.add_argument('--offcenter_ratio_th', type=float, default=0.20)
    parser.add_argument('--anchor_attack_update_th', type=float, default=0.30)
    parser.add_argument('--bg_score_th', type=float, default=0.40)
    parser.add_argument('--bg_radius', type=float, default=2.0)
    parser.add_argument('--score_drop_th', type=float, default=-0.40,
                        help='Score derivative threshold. Trigger when current-prev < this value.')
    parser.add_argument('--fast_fail_window', type=int, default=5,
                        help='Frames after a score drop to check for immediate lost transition.')
    parser.add_argument('--bbox_probe_score_th', type=float, default=0.50,
                        help='Only probe bbox area jumps when max_score is above this threshold.')
    parser.add_argument('--bbox_area_ratio_high', type=float, default=1.50,
                        help='Upper area-ratio threshold for bbox collapse probe.')
    parser.add_argument('--bbox_area_ratio_low', type=float, default=0.50,
                        help='Lower area-ratio threshold for bbox collapse probe.')

    args = parser.parse_args()

    diag_root = args.diag_root
    out_path = args.output or os.path.join(diag_root, 'auto_cause_summary.json')

    seq_dirs = _find_sequence_dirs(diag_root)
    results = []

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        csv_path = os.path.join(seq_dir, 'diag_timeline.csv')
        rows = _read_diag_csv(csv_path)
        analysis = _analyze_one(rows, args)
        entry = {
            'sequence': seq_name,
            'csv': csv_path,
        }
        entry.update(analysis)
        results.append(entry)

    counts = {
        'A_SCALE_COLLAPSE': 0,
        'B_ANCHOR_POISONING': 0,
        'WEAK_ANCHOR_POISONING': 0,
        'MIXED': 0,
        'FAST_MOTION': 0,
        'BBOX_COLLAPSE': 0,
        'FAST_MOTION_BBOX_COLLAPSE': 0,
        'UNKNOWN': 0,
    }
    for r in results:
        counts[r['label']] = counts.get(r['label'], 0) + 1

    payload = {
        'diag_root': diag_root,
        'num_sequences': len(results),
        'counts': counts,
        'results': results,
    }

    with open(out_path, 'w') as fp:
        json.dump(payload, fp, indent=2)

    print('Saved:', out_path)
    print('Counts:', counts)


if __name__ == '__main__':
    main()

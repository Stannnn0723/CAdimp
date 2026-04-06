#!/usr/bin/env python3
"""
Visualize the curvature (H_diag_ema) heatmap over time for CurvatureAwareDiMP.

Design: No Tracker / run_sequence framework — we instantiate DiMP directly,
hook update_classifier via monkeypatching, and drive the loop ourselves.
This avoids all the visdom/matplotlib/LaTOT-init_bbox bugs in the eval framework.

Usage:
  cd /data/lyx/project/pytracking-master/pytracking

  /data/lyx/anaconda3/envs/mcitrack/bin/python visualize_H_diag.py -s car1 -o ../curvature_viz -m 0.1

  # Baseline (no curvature regularization)
  /data/lyx/anaconda3/envs/mcitrack/bin/python visualize_H_diag.py -s car1 -o ../curvature_viz --baseline

  Programmatic:
  /data/lyx/anaconda3/envs/mcitrack/bin/python -c "
  from visualize_H_diag import run_sequence
  run_sequence('car1', use_curvature_reg=True, curvature_ema_momentum=0.1)
  "
"""

import os
import sys
import argparse
import copy
import numpy as np
import scipy.ndimage as _scipy_ndimage  # for zoom in alignment diagnosis

# MUST be before any pytracking/torch imports
import torch
import torch.serialization
# Patch torch.load to disable weights_only in PyTorch 2.6+ (not needed for torch<2.6)
_torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
if _torch_version >= (2, 6):
    torch.serialization._orig_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return torch.serialization._orig_load(*args, **kwargs)
    torch.load = _safe_load

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.insert(0, env_path)

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytracking.parameter.dimp.super_dimp import parameters as super_dimp_params
from pytracking.tracker.dimp.dimp import DiMP


# ─────────────────────────────────────────────────────────────────────────────
#   Profiling state — injected into DiMP.update_classifier at runtime
# ─────────────────────────────────────────────────────────────────────────────

_profiler = {
    'H_diag_history': [],
    'frame_nums': [],
    'update_nums': [],
    'scores_max': [],
    # [NEW] Per-update alignment data for H_diag validation
    'search_patches': [],    # (1,3,H,W) uint8 search image crop
    'score_maps': [],        # (1,1,H,W) raw score map at update time
    'target_masks': [],      # (1,1,H,W) target_mask from optimizer
    'gt_bboxes': [],         # (4,) [x,y,w,h] ground truth in image coords
    'pred_bboxes': [],       # (4,) [x,y,w,h] predicted bbox in image coords
    'H_diag_raw': [],        # raw (un-EMA'd) H_diag for this update
    'enabled': False,
}


def _make_profiler_wrapper(orig_update_classifier):
    """Returns a patched update_classifier that captures H_diag_ema on each call.

    [NEW] Also captures the raw score map, search crop, and bounding boxes
    so we can later overlay H_diag on the actual image to diagnose alignment.
    """
    def profiler_wrapper(self, train_x, target_box, learning_rate=None, scores=None):
        out = orig_update_classifier(self, train_x, target_box, learning_rate, scores)
        if _profiler['enabled']:
            opt = self.net.classifier.filter_optimizer
            if opt.use_curvature_reg and opt.H_diag_ema is not None:
                _profiler['H_diag_history'].append(opt.H_diag_ema.detach().cpu().clone())
                _profiler['frame_nums'].append(self.frame_num)
                _profiler['update_nums'].append(len(_profiler['H_diag_history']) - 1)
                # scores_max
                if scores is not None:
                    try:
                        _profiler['scores_max'].append(float(scores.max().detach().cpu()))
                    except Exception:
                        _profiler['scores_max'].append(float('nan'))

                # [NEW] Capture score map, search crop, and bboxes
                # scores: (N_scale, 1, H_s, W_s) — pick the same scale_ind used in track()
                # We store the full tensor; consumer can pick scale 0 (the most common one).
                if scores is not None:
                    _profiler['score_maps'].append(scores.detach().cpu().clone())
                else:
                    _profiler['score_maps'].append(torch.zeros(1, 1, 16, 16))

                # target_mask from optimizer (computed inside filter_optimizer.forward)
                # It's only available if we reach inside the optimizer; we approximate
                # by reading self.target_mask if the optimizer exposes it.
                # Fallback: store zeros so the consumer can skip it.
                tm = getattr(opt, '_last_target_mask', None)
                if tm is not None:
                    _profiler['target_masks'].append(tm.detach().cpu().clone())
                else:
                    _profiler['target_masks'].append(torch.zeros_like(_profiler['score_maps'][-1]))

                # Search crop: reconstruct from stored im_patches / sample_coords
                # DiMP stores self._last_sample_coords and self._last_im_patches
                sc = getattr(self, '_last_sample_coords', None)
                ip = getattr(self, '_last_im_patches', None)
                if sc is not None and ip is not None:
                    # sc: (N_scale, 4) [y1, x1, y2, x2] in full-image coords
                    # ip: (N_scale, 3, H_crop, W_crop)
                    _profiler['search_patches'].append(ip[0].cpu().clone())  # store first-scale crop
                else:
                    # placeholders — consumer must handle None
                    _profiler['search_patches'].append(None)

                # Ground-truth bbox at this frame (from info dict if available)
                gt = getattr(self, '_profiler_gt_bbox', None)
                _profiler['gt_bboxes'].append(gt.copy() if gt is not None else [0, 0, 0, 0])

                # Predicted bbox
                if hasattr(self, 'pos') and hasattr(self, 'target_sz'):
                    x, y = float(self.pos[0].item()), float(self.pos[1].item())
                    w, h = float(self.target_sz[0].item()), float(self.target_sz[1].item())
                    _profiler['pred_bboxes'].append([x - w/2, y - h/2, w, h])
                else:
                    _profiler['pred_bboxes'].append([0, 0, 0, 0])

                # Raw (non-EMA'd) H_diag computed this frame
                raw_H = getattr(opt, '_last_H_diag_raw', None)
                if raw_H is not None:
                    _profiler['H_diag_raw'].append(raw_H.detach().cpu().clone())
                else:
                    _profiler['H_diag_raw'].append(opt.H_diag_ema.detach().cpu().clone())
        return out
    return profiler_wrapper


# ─────────────────────────────────────────────────────────────────────────────
#   Image loader (minimal, no jpeg4py needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path):
    """Load a JPEG/PNG as an np.uint8 RGB array (H, W, 3)."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # (H, W, 3) np.uint8
    return img


# ─────────────────────────────────────────────────────────────────────────────
#   Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_top_channel_evolution(H_list, frame_nums, n_snaps=6, top_k=12, title='', out_path=''):
    """Key paper figure: top-K channels × time snapshots, spatial K_h×K_w per cell."""
    if not H_list:
        return

    last_H = H_list[-1]          # (NS, D, K_h, K_w)
    ch_means = last_H.mean(dim=(0, 2, 3))   # (D,)
    top_ch = torch.topk(ch_means, min(top_k, last_H.shape[1])).indices.tolist()

    snap_idx = np.linspace(0, len(H_list) - 1, n_snaps, dtype=int)
    ncols, nrows = n_snaps, len(top_ch)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    vmax = last_H.max().item()

    for row, ch in enumerate(top_ch):
        for col, idx in enumerate(snap_idx):
            H_spatial = H_list[idx][:, ch].mean(dim=0).numpy()   # (K_h, K_w)
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.imshow(H_spatial, cmap='hot', vmin=0, vmax=vmax)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Upd {idx+1}', fontsize=7)
            if col == ncols - 1:
                fig.colorbar(plt.cm.ScalarMappable(
                    norm=plt.Normalize(0, vmax), cmap='hot'), ax=ax, shrink=0.7)

    for row, ch in enumerate(top_ch):
        ax = axes[row, 0] if nrows > 1 else axes[0]
        ax.annotate(f'ch{ch}', xy=(-0.02, 0.5), xycoords='axes fraction',
                    fontsize=7, ha='right', va='center', rotation=90)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_spatial_evolution(H_list, frame_nums, n_snaps=6, title='', out_path=''):
    """Channel-averaged spatial H map at n snapshots."""
    if not H_list:
        return
    snap_idx = np.linspace(0, len(H_list) - 1, n_snaps, dtype=int)
    spatial = [H_list[i].mean(dim=(0, 1)).numpy() for i in snap_idx]
    vmax = max(m.max() for m in spatial)

    fig, axes = plt.subplots(1, n_snaps, figsize=(n_snaps * 2.5, 2.5))
    if n_snaps == 1:
        axes = [axes]
    for ax, smap, idx in zip(axes, spatial, snap_idx):
        im = ax.imshow(smap, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'Upd {idx+1}\nFr {frame_nums[idx]}', fontsize=7)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_timecourse(H_list, update_nums, scores_max, title='', out_path=''):
    """Mean H_diag over time with optional max-score overlay."""
    if not H_list:
        return
    mean_H = [H.mean().item() for H in H_list]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(update_nums, mean_H, 'o-', lw=1.5, ms=3, color='tab:red', label='Mean H_diag')
    ax.set_xlabel('Update number')
    ax.set_ylabel('H_diag (mean over all channels & spatial)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if scores_max:
        ax2 = ax.twinx()
        ax2.plot(update_nums, scores_max, 's--', lw=1, ms=2,
                 color='tab:blue', alpha=0.5, label='Max score')
        ax2.set_ylabel('Max score', color='tab:blue')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_first_vs_last_change(H_list, title='', out_path=''):
    """First vs last update + ΔH spatial map."""
    if len(H_list) < 2:
        return
    first = H_list[0].mean(dim=(0, 1)).numpy()   # (K_h, K_w)
    last  = H_list[-1].mean(dim=(0, 1)).numpy()
    diff  = last - first
    vmax  = max(first.max(), last.max())

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, data, label, vm, vM in zip(
        axes,
        [first, last, diff],
        ['First update', 'Last update', 'Δ (last − first)'],
        [0, 0, None],
        [vmax, vmax, None]
    ):
        im = ax.imshow(data, cmap='hot', vmin=vm, vmax=vM)
        ax.set_title(label, fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_last_grid(H_last, top_k=16, title='', out_path=''):
    """Per-channel spatial grid for the last update."""
    # H_last: (NS, D, K_h, K_w) or (D, K_h, K_w) → squeeze to (D, K_h, K_w)
    H = H_last
    while H.dim() > 3:
        H = H.mean(dim=0)
    D, K_h, K_w = H.shape
    ch_means = H.mean(dim=(1, 2))
    top_ch = torch.topk(ch_means, min(top_k, D)).indices.tolist()
    H_plot = H[top_ch]
    D_p = len(top_ch)
    ncols = min(8, D_p)
    nrows = (D_p + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axes = np.atleast_2d(axes)
    vmax = H_plot.max().item()
    for d, ch in enumerate(top_ch):
        row, col = d // ncols, d % ncols
        ax = axes[row, col]
        ax.imshow(H[d].numpy(), cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'ch{ch}', fontsize=6)
        ax.axis('off')
        if col == ncols - 1:
            fig.colorbar(plt.cm.ScalarMappable(
                norm=plt.Normalize(0, vmax), cmap='hot'), ax=ax, shrink=0.7)
    for d in range(D_p, nrows * ncols):
        axes[d // ncols, d % ncols].axis('off')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#   [NEW] H_diag Alignment Diagnosis — the core verification figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_H_diag_alignment_diagnosis(
    H_list, raw_H_list, score_maps, search_patches, gt_bboxes, pred_bboxes,
    frame_nums, update_nums, n_snaps=None, title='', out_dir=''):
    os.makedirs(out_dir, exist_ok=True)
    n = len(H_list)
    if n == 0:
        return
    snap_idx = np.linspace(0, n - 1, min(n_snaps, n) if n_snaps else n, dtype=int).tolist()

    # Per-update detailed PNGs
    for idx in snap_idx:
        fig = _render_single_diagnosis_frame(
            H_list[idx],
            raw_H_list[idx] if idx < len(raw_H_list) else H_list[idx],
            score_maps[idx] if idx < len(score_maps) else None,
            search_patches[idx] if (idx < len(search_patches) and search_patches[idx] is not None) else None,
            gt_bboxes[idx] if idx < len(gt_bboxes) else None,
            pred_bboxes[idx] if idx < len(pred_bboxes) else None,
            frame_nums[idx], update_nums[idx],
        )
        fig.savefig(os.path.join(out_dir, f'align_upd{idx:03d}_fr{frame_nums[idx]:04d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f"  ✓ wrote {len(snap_idx)} per-update alignment PNGs")

    # Montage: [search | score | H_raw | H_ema] × n_snaps
    ncols, nrows = 4, len(snap_idx)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for row, idx in enumerate(snap_idx):
        _fill_diagnosis_row(
            axes[row], H_list[idx],
            raw_H_list[idx] if idx < len(raw_H_list) else H_list[idx],
            score_maps[idx] if idx < len(score_maps) else None,
            search_patches[idx] if (idx < len(search_patches) and search_patches[idx] is not None) else None,
            gt_bboxes[idx] if idx < len(gt_bboxes) else None,
            frame_nums[idx], update_nums[idx],
        )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    montage_path = os.path.join(out_dir, 'alignment_montage.png')
    fig.savefig(montage_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ alignment_montage.png")


def _render_single_diagnosis_frame(H_ema, H_raw, score_map, search_patch,
                                   gt_bbox, pred_bbox, frame_num, update_num):
    """One detailed multi-panel figure for a single update."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(f'Update {update_num}  |  Frame {frame_num}', fontsize=11, fontweight='bold')

    def _draw_heatmap(data, ax, label, vmax=None):
        ax.clear(); ax.set_title(label, fontsize=8); ax.axis('off')
        d = data
        while d.dim() > 2:
            d = d.mean(dim=0)
        d = d.numpy()
        if d.ndim == 2:
            vm = d.max() if vmax is None else vmax
            im = ax.imshow(d, cmap='hot', vmin=0, vmax=vm)
            plt.colorbar(im, ax=ax, shrink=0.6)

    # Row 1
    if search_patch is not None:
        axes[0, 0].clear()
        axes[0, 0].set_title('Search Crop + GT/Pred', fontsize=8); axes[0, 0].axis('off')
        img = search_patch.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        axes[0, 0].imshow(img)
        if gt_bbox is not None:
            x, y, w, h = gt_bbox
            axes[0, 0].add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                                edgecolor='lime', linewidth=1.5))
            axes[0, 0].text(x, max(y-2, 5), 'GT', color='lime', fontsize=7)
        if pred_bbox is not None:
            xp, yp, wp, hp = pred_bbox
            axes[0, 0].add_patch(plt.Rectangle((xp, yp), wp, hp, fill=False,
                                                edgecolor='cyan', linewidth=1.5))
            axes[0, 0].text(xp, max(yp-2, 5), 'Pred', color='cyan', fontsize=7)
    else:
        axes[0, 0].clear(); axes[0, 0].text(0.5, 0.5, 'No crop', ha='center', va='center')
        axes[0, 0].axis('off')

    _draw_heatmap(score_map if score_map is not None else torch.zeros(1,1,4,4),
                  axes[0, 1], 'Score Map')
    _draw_heatmap(H_raw, axes[0, 2], 'H_diag raw (feat space)')
    _draw_heatmap(H_ema, axes[1, 0], 'H_diag EMA (feat space)')

    # Row 2, col 1: scatter — is high H correlated with high score?
    ax = axes[1, 1]; ax.clear(); ax.set_title('Score vs H corr', fontsize=8); ax.axis('off')
    if score_map is not None and H_ema is not None:
        try:
            s = score_map.squeeze().cpu().numpy()
            H_ch = H_ema.mean(dim=(0,1)).numpy()  # (K_h, K_w)
            zh, zw = s.shape[0]/H_ch.shape[0], s.shape[1]/H_ch.shape[1]
            H_up = _scipy_ndimage.zoom(H_ch, (zh, zw), order=1)
            s_n = (s - s.min()) / (s.max() - s.min() + 1e-8)
            H_n = (H_up - H_up.min()) / (H_up.max() - H_up.min() + 1e-8)
            ax.axis('on')
            ax.scatter(H_n.ravel(), s_n.ravel(), s=0.5, alpha=0.3)
            ax.set_xlabel('H_ema (norm)'); ax.set_ylabel('Score (norm)')
            corr = np.corrcoef(H_n.ravel(), s_n.ravel())[0, 1]
            ax.set_title(f'Score vs H corr={corr:.3f}', fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, 'corr failed', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # Row 2, col 2: centroid overlay
    ax = axes[1, 2]; ax.clear(); ax.set_title('H vs Score peak', fontsize=8); ax.axis('off')
    if score_map is not None and H_ema is not None:
        try:
            s = score_map.squeeze().cpu().numpy()
            H_ch = H_ema.mean(dim=(0,1)).numpy()
            score_h, score_w = s.shape
            H_h, H_w = H_ch.shape
            gy_s, gx_s = np.unravel_index(s.argmax(), s.shape)
            gy_H, gx_H = np.unravel_index(H_ch.argmax(), H_ch.shape)
            gx_Hn = gx_H / H_w * score_w
            gy_Hn = gy_H / H_h * score_h
            ax.axis('on')
            ax.imshow(s, cmap='gray', alpha=0.3)
            H_up = _scipy_ndimage.zoom(H_ch, (score_h/H_h, score_w/H_w), order=1)
            overlay = np.zeros((score_h, score_w, 4))
            overlay[..., 0] = 1.0; overlay[..., 3] = H_up / (H_up.max()+1e-8) * 0.6
            ax.imshow(overlay)
            ax.plot(gx_s, gy_s, 'c+', ms=10, mew=2, label='Score peak')
            ax.plot(gx_Hn, gy_Hn, 'rX', ms=10, mew=2, label='H peak')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_title(f'Centroid Δ=({gx_s-gx_Hn:.1f},{gy_s-gy_Hn:.1f})px', fontsize=7)
        except Exception:
            ax.text(0.5, 0.5, 'overlay failed', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
    return fig


def _fill_diagnosis_row(axes_row, H_ema, H_raw, score_map, search_patch, gt_bbox, frame_num, update_num):
    """Fill one montage row: [search | score | H_raw | H_ema]."""
    items = [
        (search_patch,   True,  'Search + GT'),
        (score_map,      False, 'Score Map'),
        (H_raw,          False, 'H_raw'),
        (H_ema,          False, 'H_EMA'),
    ]
    for col, (data, is_rgb, label) in enumerate(items):
        ax = axes_row[col]
        ax.clear()
        ax.set_title(f'{label}\nU{update_num}/F{frame_num}', fontsize=7)
        ax.axis('off')
        if data is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            continue
        if is_rgb:
            img = data.permute(1, 2, 0).numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            ax.imshow(img)
            if gt_bbox is not None:
                x, y, w, h = gt_bbox
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                           edgecolor='lime', linewidth=1.5))
        else:
            d = data
            while d.dim() > 2:
                d = d.mean(dim=0)
            d_np = d.numpy() if isinstance(d, torch.Tensor) else d
            if d_np.ndim == 2:
                im = ax.imshow(d_np, cmap='hot')
                plt.colorbar(im, ax=ax, shrink=0.5)
            else:
                ax.text(0.5, 0.5, f'{data.shape}', ha='center', va='center', fontsize=7)

def run_sequence(seq_name,
                 use_curvature_reg=True,
                 curvature_ema_momentum=0.5,
                 curvature_anchor_interval=1,
                 curvature_reg_weight=10.0,
                 output_dir='../curvature_viz',
                 debug=0):
    """
    Run DiMP on one LaTOT sequence with H_diag profiling.

    Returns:
        dict with 'H_diag' (tensor), 'frame_nums', 'update_nums', 'scores_max'
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seq_dir = os.path.join(output_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)

    # ── Reset profiler ───────────────────────────────────────────────────
    for k in _profiler:
        if isinstance(_profiler[k], list):
            _profiler[k].clear()
    _profiler['enabled'] = False

    # ── Build params ───────────────────────────────────────────────────────
    params = super_dimp_params()
    params.use_curvature_reg = use_curvature_reg
    params.curvature_ema_momentum = curvature_ema_momentum
    params.curvature_anchor_interval = curvature_anchor_interval
    params.curvature_reg_weight = curvature_reg_weight
    params.debug = debug
    params.visualization = False
    params.use_gpu = True   # GPU available

    if debug >= 1:
        tag = 'CurvatureAwareDiMP' if use_curvature_reg else 'Baseline DiMP'
        print(f"[{tag}]  β={curvature_ema_momentum}  interval={curvature_anchor_interval}  λ={curvature_reg_weight}")

    # ── Load sequence ───────────────────────────────────────────────────
    from pytracking.evaluation import get_dataset
    dataset = get_dataset('latot')
    try:
        seq = dataset[seq_name]
    except Exception as e:
        print(f"[ERROR] Sequence '{seq_name}' not in LaTOT: {e}")
        return None

    if debug >= 1:
        print(f"  Sequence: {seq_name}  frames: {len(seq.frames)}")

    # ── Patch DiMP.update_classifier BEFORE any tracking ─────────────────
    if not hasattr(DiMP, '_orig_update_classifier'):
        DiMP._orig_update_classifier = DiMP.update_classifier
        DiMP.update_classifier = _make_profiler_wrapper(DiMP._orig_update_classifier)

    _profiler['enabled'] = True

    # ── Create tracker ───────────────────────────────────────────────────
    tracker = DiMP(params)
    tracker.params = params      # ensure our params are used
    tracker.params.net.use_gpu = True   # GPU available
    tracker.initialize_features()

    # ── Get init bbox from LaTOT ground truth ────────────────────────────
    gt = seq.ground_truth_rect          # (N, 4) numpy array [x, y, w, h]
    init_bbox = copy.deepcopy(gt[0])     # [x, y, w, h] as numpy array

    # ── Pass full GT timeline into the tracker ──────────────────────────
    tracker._profiler_gt_timeline = gt.copy()   # (N, 4) [x, y, w, h]

    # ── Load first image ────────────────────────────────────────────────
    first_img = _load_image(seq.frames[0])
    init_info = {'init_bbox': init_bbox}
    tracker.initialize(first_img, init_info)

    if debug >= 1:
        print(f"  Initialized.  Training samples stored: {getattr(tracker, 'num_stored_samples', '?')}")

    # ── Track remaining frames ───────────────────────────────────────────
    bboxes = [init_bbox.copy()]
    for frame_idx, frame_path in enumerate(seq.frames[1:], start=1):
        img = _load_image(frame_path)
        info = {'previous_output': {'target_bbox': bboxes[-1]}}

        # [NEW] Feed GT bbox for this frame to the profiler
        tracker._profiler_gt_bbox = gt[frame_idx].tolist()

        out = tracker.track(img, info)
        bboxes.append(out['target_bbox'])

        if debug >= 1 and (frame_idx % 50 == 0 or frame_idx == len(seq.frames) - 1):
            print(f"  Frame {frame_idx}/{len(seq.frames)-1}  bbox={bboxes[-1]}")

    # ── Disable profiler ─────────────────────────────────────────────────
    _profiler['enabled'] = False

    n_upd = len(_profiler['H_diag_history'])
    if debug >= 1:
        print(f"  → Captured {n_upd} H_diag updates")

    if n_upd == 0:
        print("[WARN] No H_diag captured. Is use_curvature_reg=True?")
        return None

    # ── Snapshot data before profiler is reused ───────────────────────────
    H_snapshot = [_h.clone() for _h in _profiler['H_diag_history']]
    H_raw_snapshot = [_h.clone() for _h in _profiler.get('H_diag_raw', [])]
    score_map_snapshot = [_s.clone() for _s in _profiler.get('score_maps', [])]
    search_patch_snapshot = list(_profiler.get('search_patches', []))
    gt_snapshot = list(_profiler.get('gt_bboxes', []))
    pred_snapshot = list(_profiler.get('pred_bboxes', []))
    fn_snap = list(_profiler['frame_nums'])
    un_snap = list(_profiler['update_nums'])
    sc_snap = list(_profiler['scores_max'])
    H_traj  = torch.stack(H_snapshot)    # (n_upd, NS, D, K_h, K_w)

    prefix = f"{seq_name}_curv" if use_curvature_reg else f"{seq_name}_base"
    tag    = f"β={curvature_ema_momentum} λ={curvature_reg_weight}"

    # ── Save figures ───────────────────────────────────────────────────
    # 1. Top-K channel × time evolution (论文主图)
    _plot_top_channel_evolution(
        H_snapshot, fn_snap, n_snaps=6, top_k=12,
        title=f'{seq_name} — Top-12 channel spatial H evolution ({tag})',
        out_path=os.path.join(seq_dir, f'{prefix}_top_channels.png'),
    )
    print(f"  ✓ {prefix}_top_channels.png")

    # 2. Channel-averaged spatial evolution
    _plot_spatial_evolution(
        H_snapshot, fn_snap, n_snaps=6,
        title=f'{seq_name} — Channel-averaged spatial H evolution ({tag})',
        out_path=os.path.join(seq_dir, f'{prefix}_spatial_evo.png'),
    )
    print(f"  ✓ {prefix}_spatial_evo.png")

    # 3. Mean H timecourse
    _plot_timecourse(
        H_snapshot, un_snap, sc_snap,
        title=f'{seq_name} — Mean H_diag over updates ({tag})',
        out_path=os.path.join(seq_dir, f'{prefix}_timecourse.png'),
    )
    print(f"  ✓ {prefix}_timecourse.png")

    # 4. First vs last + Δ map
    _plot_first_vs_last_change(
        H_snapshot,
        title=f'{seq_name} — H_diag spatial change ({tag})',
        out_path=os.path.join(seq_dir, f'{prefix}_change.png'),
    )
    print(f"  ✓ {prefix}_change.png")

    # 5. Last-update per-channel grid
    _plot_last_grid(
        H_snapshot[-1], top_k=16,
        title=f'{seq_name} — Per-channel H (last update, {tag})',
        out_path=os.path.join(seq_dir, f'{prefix}_last_grid.png'),
    )
    print(f"  ✓ {prefix}_last_grid.png")

    # 6. [NEW] H_diag alignment diagnosis — the core verification figure
    _plot_H_diag_alignment_diagnosis(
        H_list=H_snapshot,
        raw_H_list=H_raw_snapshot,
        score_maps=score_map_snapshot,
        search_patches=search_patch_snapshot,
        gt_bboxes=gt_snapshot,
        pred_bboxes=pred_snapshot,
        frame_nums=fn_snap,
        update_nums=un_snap,
        n_snaps=8,
        title=f'{seq_name} — H_diag Alignment Diagnosis ({tag})',
        out_dir=os.path.join(seq_dir, 'alignment_diagnosis'),
    )

    # 7. Raw trajectory
    npz = os.path.join(seq_dir, f'{prefix}_traj.npz')
    np.savez_compressed(npz,
                       H_diag=H_traj.numpy(),
                       frame_nums=np.array(fn_snap),
                       update_nums=np.array(un_snap),
                       scores_max=np.array(sc_snap))
    print(f"  ✓ {prefix}_traj.npz  ({n_upd} updates, H_traj shape: {tuple(H_traj.shape)})")

    print(f"\nAll outputs → {seq_dir}/")
    return {
        'H_diag': H_traj,
        'frame_nums': fn_snap,
        'update_nums': un_snap,
        'scores_max': sc_snap,
        'bboxes': np.array(bboxes),
        'search_patches': search_patch_snapshot,
        'score_maps': score_map_snapshot,
        'gt_bboxes': gt_snapshot,
        'pred_bboxes': pred_snapshot,
    }


# ─────────────────────────────────────────────────────────────────────────────
#   CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Visualize H_diag curvature heatmaps')
    p.add_argument('-s', '--sequence', default='car1')
    p.add_argument('-o', '--output_dir', default='../curvature_viz')
    p.add_argument('-m', '--momentum', type=float, default=0.5,
                   help='EMA momentum β (larger = more visible changes, 0.5 recommended for diagnosis)')
    p.add_argument('-i', '--interval', type=int, default=1,
                   help='Anchor update interval (1 = every frame, maximises H temporal sensitivity)')
    p.add_argument('-w', '--reg_weight', type=float, default=10.0,
                   help='Curvature reg weight multiplier (default 10.0, baseline 1.0)')
    p.add_argument('--baseline', action='store_true',
                   help='Baseline DiMP (no curvature regularization)')
    p.add_argument('-d', '--debug', type=int, default=0,
                   help='0=silent 1=progress')
    args = p.parse_args()
    run_sequence(
        seq_name=args.sequence,
        use_curvature_reg=(not args.baseline),
        curvature_ema_momentum=args.momentum,
        curvature_anchor_interval=args.interval,
        curvature_reg_weight=args.reg_weight,
        output_dir=args.output_dir,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()

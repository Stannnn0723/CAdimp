"""DiMP tracker params for LaTOT continued-training (super_dimp_latot) checkpoints.

Same tracking / curvature-aware runtime settings as super_dimp.py. Point net_path to
your fine-tuned checkpoint: either symlink/copy ``*_epXXXX.pth.tar`` to
``pytracking/networks/super_dimp_latot_ft.pth.tar``, or set an absolute path below.
"""
from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 22*16
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # [CURVATURE-AWARE] Curvature-Aware Anisotropic Regularization hyperparameters.
    # Set use_curvature_reg=True to enable.  No retraining required — works with
    # the pre-trained super_dimp.pth.tar checkpoint injected at runtime.
    params.use_curvature_reg = True  # 改为 False 关闭曲率感知（原版 Super DiMP）
    # EMA momentum for the quasi-Hessian diagonal accumulation (H_diag_ema).
    # Smaller values → smoother, slower adaptation; larger values → faster response
    # to new background configurations but more noise.  Typical range: [0.005, 0.05].
    params.curvature_ema_momentum = 0.01
    # Only update the slow anchor weights (weights_anchor) every N normal updates.
    # Set > 1 to prevent distractor-corrupted frames from polluting the anchor.
    params.curvature_anchor_interval = 10
    # Anchor update strategy:
    # 'smooth': EMA update every optimizer call (current default behavior).
    # 'step':   hold for interval calls, then hard copy update.
    # 'hybrid': hold for interval calls, then EMA with curvature_anchor_step_beta.
    params.curvature_anchor_update_mode = 'step'
    # EMA factor used only when curvature_anchor_update_mode='hybrid'.
    params.curvature_anchor_step_beta = 0.05
    # Where to initialise the slow anchor: 'init' (anchor from first high-quality
    # frame) or 'current' (anchor from the most recent clean frame).
    params.curvature_init_anchor_from = 'init'
    # Multiplier on the anisotropic curvature term (on top of filter_reg=1.0).
    # - 1.0  = same scale as isotropic L2 reg  (safe baseline, ~0 effect)
    # - 5–20 = meaningful memory shield on high-curvature directions.
    # Recommended: start at 10.0 for LaTOT/UAV123 experiments.
    params.curvature_reg_weight = 10.0

    # [STATE-AWARE SHIELD] Master switch for the shield mechanism.
    # When True: cur_weight switches between soft (1.0) and hard (10.0) based on
    # the current max classification score and peripheral distractor detection.
    # When False: cur_weight is always curv_soft_weight (1.0), i.e. shield is disabled.
    params.use_curvature_shield = True
    # [SOFT GATING] Toggle for the smooth sigmoid transition of the shield.
    # When True: penalty weight increases linearly using a Sigmoid function instead of a hard step.
    params.use_curv_soft_gating = False
    # Threshold for switching to hard (heavy-shield) mode.
    # When max_score < curv_low_score_th → λ = curv_hard_weight (10.0).
    # When max_score >= curv_low_score_th → λ = curv_soft_weight (1.0).
    # Tunable range: [0.2, 0.5].  Lower → more defensive; higher → more adaptive.
    params.curv_low_score_th = 0.35
    # Weight for the hard/shield mode (distractor / occlusion / low-confidence frames).
    params.curv_hard_weight = 10.0
    # Weight for the soft/free mode (normal tracking frames — allows filter adaptation).
    params.curv_soft_weight = 1.0
    # [DISTRACTOR-AWARE RADAR] Threshold on the peripheral (off-centre) score map.
    # If the max score OUTSIDE the centre region exceeds this → heavy shield ON.
    # Tunable range: [0.3, 0.6].  Lower → more sensitive to background noise.
    params.curv_bg_thresh = 0.40
    # Half-width of the centre "target zone" to mask out (in feature-map pixels).
    # Target occupies roughly ±2–3 px in feature-map space (stride=16 → ±3 in map).
    params.curv_bg_radius = 2

    params.curvature_tg_ratio_thresh = 0.5
    # [CHANNEL RELIABILITY] Spatial variance based channel quality filter.
    # When True: H_diag is weighted by per-channel reliability derived from spatial compactness.
    # When False: use raw H_diag (baseline).
    params.use_channel_reliability = True
    # Strength of the spread penalty (spread → reliability mapping).
    # High value → aggressive suppression of high-spread channels (background semantics).
    # Recommended range: [1.0, 5.0].  Start at 2.0.
    params.channel_reliability_strength = 2.0
    # Spread threshold in normalized feature-map coordinates: channels with
    # spread > thresh are treated as unreliable.
    # Recommended range: [0.5, 2.0].  Start at 1.0.
    params.channel_spread_thresh = 1.0

    # [DIAGNOSTIC LOGGER] Failure-case state dump switch.
    # Set True to enable per-frame internal-state logging and failed-sequence overlays.
    # Set False to disable saving failure cases and avoid performance/storage overhead during large scale testing.
    params.diag_debug_mode = False
    # A sequence is archived as failure when mean IoU < this threshold.
    params.diag_failure_iou_thresh = 0.30
    # Export visual overlays (raw frame + heatmap + internal state text) for failures.
    params.diag_dump_visual_overlay = False
    # Cap number of exported frames per failed sequence (lowest-IoU frames first).
    params.diag_dump_max_frames = 60

    # [CURVATURE ANALYSIS] Quantitative statistics for GT overlap on LaTOT.
    # Generates per-sequence csv/json and dataset_summary under results/.../curvature_analysis/.
    params.enable_curvature_analysis = False
    # Top-k ratio used for high-curvature hit-rate (e.g. top 5% pixels in score-map grid).
    params.curvature_analysis_topk_ratio = 0.05

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                          'rotate': [10, -10, 45, -45],
                          'blur': [(3,1), (1, 3), (2, 2)],
                          'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                          'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    # Default: place your LTR checkpoint (or symlink) under pytracking/networks/.
    # Alternative: pass an absolute path, e.g. '/path/to/workspace/checkpoints/ltr/dimp/super_dimp_latot/SuperDiMP_ep0050.pth.tar'
    params.net = NetWithBackbone(net_path='/data/lyx/project/pytracking-master/checkpoints/ltr/dimp/super_dimp_latot/DiMPnet_ep0050.pth.tar',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    return params

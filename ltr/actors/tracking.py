from . import BaseActor
import math
import torch
import torch.nn.functional as F


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None, curvature_params=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.curvature_params = curvature_params if curvature_params is not None else {}

    def _get_compact_warmup_weight(self, settings, epoch):
        if not getattr(settings, 'use_curvature_compact_loss', False):
            return 0.0

        final_weight = float(getattr(settings, 'curvature_compact_loss_weight', 0.0))
        start_epoch = int(getattr(settings, 'curvature_compact_warmup_start_epoch', 0))
        end_epoch = int(getattr(settings, 'curvature_compact_warmup_end_epoch', start_epoch))
        warmup_type = getattr(settings, 'curvature_compact_warmup_type', 'cosine')

        if epoch <= start_epoch:
            return 0.0
        if end_epoch <= start_epoch or epoch >= end_epoch:
            return final_weight

        progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)
        progress = max(0.0, min(1.0, progress))

        if warmup_type == 'cosine':
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            scale = progress
        return final_weight * scale

    def _compute_compact_loss(self, data, aux_data):
        settings = data['settings']
        eps = float(getattr(settings, 'curvature_compact_eps', 1e-6))

        feat_source = getattr(settings, 'curvature_compact_feat_source', 'test')
        if feat_source == 'train':
            feat = aux_data.get('train_feat_clf', None)
            anno = data['train_anno']
        else:
            feat = aux_data.get('test_feat_clf', None)
            anno = data['test_anno']

        label_map = aux_data.get('optimizer_label_map', None)
        if feat is None or label_map is None:
            zero = data['test_images'].new_tensor(0.0)
            return zero, {
                'selected_mean': 0.0,
                'selected_min': 0.0,
                'selected_max': 0.0,
                'empty_ratio': 1.0,
                'mean_spread_selected': 0.0,
                'mean_spread_all': 0.0,
                'mean_rtg_selected': 0.0,
                'mean_q_selected': 0.0,
                'mean_centroid_min_dist': 0.0,
                'selection_success_ratio': 0.0,
            }

        num_images, num_sequences, num_channels, feat_h, feat_w = feat.shape
        feat_flat = feat.reshape(-1, num_channels, feat_h, feat_w)

        label_h, label_w = label_map.shape[-2], label_map.shape[-1]
        if label_h != feat_h or label_w != feat_w:
            label_map = label_map[..., :feat_h, :feat_w].contiguous()

        label_flat = label_map.reshape(-1, feat_h, feat_w)
        anno_flat = anno.reshape(-1, anno.shape[-1])
        valid_flat = (anno_flat[:, 0] < 99999.0)

        core_thresh = float(getattr(settings, 'curvature_mask_core_thresh', 0.6))
        ring_thresh = float(getattr(settings, 'curvature_mask_ring_thresh', 0.1))
        out_thresh = float(getattr(settings, 'curvature_mask_out_thresh', 0.05))
        ring_weight = float(getattr(settings, 'curvature_mask_ring_weight', 0.35))

        rtg_thresh = float(getattr(settings, 'curvature_rtg_thresh', 0.6))
        select_k_min = int(getattr(settings, 'curvature_select_k_min', 8))
        select_k_max = int(getattr(settings, 'curvature_select_k_max', 24))
        select_pre_k = int(getattr(settings, 'curvature_select_pre_k', 64))
        centroid_min_dist = float(getattr(settings, 'curvature_centroid_min_dist', 0.18))

        tau = float(getattr(settings, 'curvature_compact_tau', 0.18))
        beta = float(getattr(settings, 'curvature_compact_beta', 0.05))
        use_q_weight = bool(getattr(settings, 'curvature_compact_use_q_weight', True))
        min_size_cells = float(getattr(settings, 'curvature_compact_min_size_cells', 0.0))
        use_dynamic_tau = bool(getattr(settings, 'curvature_compact_use_dynamic_tau', False))
        dynamic_tau_alpha = float(getattr(settings, 'curvature_compact_dynamic_tau_alpha', 1.0))
        dynamic_tau_area_ref = float(getattr(settings, 'curvature_compact_dynamic_tau_area_ref', 0.02))

        m_core = (label_flat > core_thresh).float()
        m_ring = ((label_flat > ring_thresh) & (label_flat <= core_thresh)).float() * ring_weight
        m_in = (m_core + m_ring).clamp(max=1.0)
        m_out = (label_flat < out_thresh).float()

        y_coords = torch.linspace(0.0, 1.0, feat_h, device=feat.device, dtype=feat.dtype).view(1, 1, feat_h, 1)
        x_coords = torch.linspace(0.0, 1.0, feat_w, device=feat.device, dtype=feat.dtype).view(1, 1, 1, feat_w)

        activation = F.relu(feat_flat) ** 2

        weighted_in = activation * m_in.unsqueeze(1)
        energy_in = weighted_in.sum(dim=(-1, -2))
        energy_all = activation.sum(dim=(-1, -2))
        energy_out = (activation * m_out.unsqueeze(1)).sum(dim=(-1, -2))

        rtg = energy_in / (energy_all + eps)
        leakage = energy_out / (energy_in + eps)

        core_energy = (activation * m_core.unsqueeze(1)).sum(dim=(-1, -2))
        core_norm = core_energy + eps
        mu_x = ((activation * m_core.unsqueeze(1)) * x_coords).sum(dim=(-1, -2)) / core_norm
        mu_y = ((activation * m_core.unsqueeze(1)) * y_coords).sum(dim=(-1, -2)) / core_norm
        centroids = torch.stack((mu_x, mu_y), dim=-1)

        in_norm = weighted_in.sum(dim=(-1, -2)) + eps
        mean_x = (weighted_in * x_coords).sum(dim=(-1, -2)) / in_norm
        mean_y = (weighted_in * y_coords).sum(dim=(-1, -2)) / in_norm
        var_x = (weighted_in * (x_coords - mean_x.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-1, -2)) / in_norm
        var_y = (weighted_in * (y_coords - mean_y.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-1, -2)) / in_norm

        widths = (anno_flat[:, 2] / float(getattr(settings, 'output_sz', feat_w))).clamp(min=eps)
        heights = (anno_flat[:, 3] / float(getattr(settings, 'output_sz', feat_h))).clamp(min=eps)

        if min_size_cells > 0.0:
            min_width = min_size_cells / float(feat_w)
            min_height = min_size_cells / float(feat_h)
            widths_safe = torch.clamp(widths, min=min_width)
            heights_safe = torch.clamp(heights, min=min_height)
        else:
            widths_safe = widths
            heights_safe = heights

        spread = var_x / (widths_safe.unsqueeze(1) ** 2 + eps) + var_y / (heights_safe.unsqueeze(1) ** 2 + eps)

        if use_dynamic_tau:
            area_safe = (widths_safe * heights_safe).detach()
            area_ref = max(dynamic_tau_area_ref, eps)
            tiny_factor = torch.exp(-area_safe / area_ref)
            dynamic_tau = tau * (1.0 + dynamic_tau_alpha * tiny_factor)
        else:
            dynamic_tau = widths_safe.new_full((widths_safe.shape[0],), tau)

        q_score = energy_in * leakage

        compact_loss_acc = feat.new_tensor(0.0)
        valid_sample_count = 0
        selected_counts = []
        preselected_counts = []
        spread_selected_vals = []
        spread_all_vals = []
        rtg_selected_vals = []
        q_selected_vals = []
        centroid_dist_vals = []
        empty_count = 0
        success_count = 0

        for idx in range(feat_flat.shape[0]):
            if not bool(valid_flat[idx]):
                continue

            spread_all_vals.append(spread[idx].mean())

            with torch.no_grad():
                candidate_mask = rtg[idx] > rtg_thresh
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                preselected_counts.append(float(candidate_indices.numel()))

                if candidate_indices.numel() == 0:
                    selected_indices = candidate_indices
                    selected_q = q_score[idx].new_zeros((0,))
                    selected_min_dist = q_score[idx].new_tensor(0.0)
                else:
                    candidate_q = q_score[idx, candidate_indices]
                    pre_k = min(select_pre_k, candidate_indices.numel())
                    top_q, top_pos = torch.topk(candidate_q, k=pre_k, largest=True, sorted=True)
                    pre_idx = candidate_indices[top_pos]
                    pre_centroids = centroids[idx, pre_idx]
                    if pre_idx.numel() > 1:
                        dist_mat = torch.cdist(pre_centroids.unsqueeze(0), pre_centroids.unsqueeze(0)).squeeze(0)
                    else:
                        dist_mat = pre_centroids.new_zeros((1, 1))

                    chosen_positions = []
                    for pos in range(pre_idx.numel()):
                        if len(chosen_positions) >= select_k_max:
                            break
                        if len(chosen_positions) == 0:
                            chosen_positions.append(pos)
                            continue
                        dists = dist_mat[pos, torch.tensor(chosen_positions, device=dist_mat.device)]
                        if torch.all(dists > centroid_min_dist):
                            chosen_positions.append(pos)

                    if len(chosen_positions) < min(select_k_min, pre_idx.numel()):
                        for pos in range(pre_idx.numel()):
                            if pos not in chosen_positions:
                                chosen_positions.append(pos)
                            if len(chosen_positions) >= min(select_k_min, pre_idx.numel()):
                                break

                    chosen_positions = chosen_positions[:min(select_k_max, pre_idx.numel())]
                    if len(chosen_positions) > 0:
                        chosen_positions_t = torch.tensor(chosen_positions, device=pre_idx.device, dtype=torch.long)
                        selected_indices = pre_idx[chosen_positions_t]
                        selected_q = top_q[chosen_positions_t]
                        if selected_indices.numel() > 1:
                            selected_centroids = centroids[idx, selected_indices]
                            selected_dist = torch.cdist(selected_centroids.unsqueeze(0), selected_centroids.unsqueeze(0)).squeeze(0)
                            selected_dist = selected_dist + torch.eye(selected_dist.shape[0], device=selected_dist.device, dtype=selected_dist.dtype) * 1e6
                            selected_min_dist = selected_dist.min(dim=1)[0].mean()
                        else:
                            selected_min_dist = top_q.new_tensor(0.0)
                    else:
                        selected_indices = pre_idx[:0]
                        selected_q = top_q[:0]
                        selected_min_dist = top_q.new_tensor(0.0)

            selected_counts.append(float(selected_indices.numel()))
            centroid_dist_vals.append(selected_min_dist)

            if selected_indices.numel() == 0:
                empty_count += 1
                continue

            success_count += 1
            valid_sample_count += 1

            selected_spread = spread[idx, selected_indices]
            selected_rtg = rtg[idx, selected_indices]
            spread_selected_vals.append(selected_spread.mean())
            rtg_selected_vals.append(selected_rtg.mean())
            q_selected_vals.append(selected_q.mean())

            distance_over_tau = F.relu(selected_spread - dynamic_tau[idx])
            compact_vec = F.smooth_l1_loss(distance_over_tau, torch.zeros_like(distance_over_tau), beta=beta, reduction='none')
            if use_q_weight and selected_q.numel() > 0:
                q_weights = selected_q / (selected_q.sum() + eps)
                compact_loss_acc = compact_loss_acc + (compact_vec * q_weights).sum()
            else:
                compact_loss_acc = compact_loss_acc + compact_vec.mean()

        if valid_sample_count > 0:
            compact_loss = compact_loss_acc / valid_sample_count
        else:
            compact_loss = feat.new_tensor(0.0)

        def _safe_mean(vals):
            if len(vals) == 0:
                return 0.0
            stacked = [v if torch.is_tensor(v) else feat.new_tensor(v) for v in vals]
            return torch.stack(stacked).mean().item()

        metrics = {
            'selected_mean': sum(selected_counts) / max(len(selected_counts), 1),
            'selected_min': min(selected_counts) if len(selected_counts) > 0 else 0.0,
            'selected_max': max(selected_counts) if len(selected_counts) > 0 else 0.0,
            'preselected_mean': sum(preselected_counts) / max(len(preselected_counts), 1),
            'empty_ratio': float(empty_count) / max(len(selected_counts), 1),
            'mean_spread_selected': _safe_mean(spread_selected_vals),
            'mean_spread_all': _safe_mean(spread_all_vals),
            'mean_rtg_selected': _safe_mean(rtg_selected_vals),
            'mean_q_selected': _safe_mean(q_selected_vals),
            'mean_centroid_min_dist': _safe_mean(centroid_dist_vals),
            'selection_success_ratio': float(success_count) / max(len(selected_counts), 1),
            'mean_dynamic_tau': dynamic_tau.mean().item(),
            'mean_width_safe': widths_safe.mean().item(),
            'mean_height_safe': heights_safe.mean().item(),
        }

        return compact_loss, metrics

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores, aux_data = self.net(train_imgs=data['train_images'],
                                                      test_imgs=data['test_images'],
                                                      train_bb=data['train_anno'],
                                                      test_proposals=data['test_proposals'],
                                                      return_aux=True)

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        compact_loss = data['test_images'].new_tensor(0.0)
        compact_loss_weighted = data['test_images'].new_tensor(0.0)
        compact_metrics = None
        compact_lambda = 0.0
        if getattr(data['settings'], 'use_curvature_compact_loss', False) and self.net.training:
            compact_loss, compact_metrics = self._compute_compact_loss(data, aux_data)
            compact_lambda = self._get_compact_warmup_weight(data['settings'], int(data.get('epoch', 0)))
            compact_loss_weighted = compact_loss * compact_lambda

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + compact_loss_weighted

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item(),
                 'Loss/compact': compact_loss.item(),
                 'Loss/compact_weighted': compact_loss_weighted.item(),
                 'CurvCompact/lambda': compact_lambda,
                 'CurvCompact/tau': float(getattr(data['settings'], 'curvature_compact_tau', 0.0)),
                 'CurvCompact/beta': float(getattr(data['settings'], 'curvature_compact_beta', 0.0))}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if compact_metrics is not None:
            stats['CurvCompact/selected_channels_mean'] = compact_metrics['selected_mean']
            stats['CurvCompact/selected_channels_min'] = compact_metrics['selected_min']
            stats['CurvCompact/selected_channels_max'] = compact_metrics['selected_max']
            stats['CurvCompact/preselected_channels_mean'] = compact_metrics['preselected_mean']
            stats['CurvCompact/empty_selection_ratio'] = compact_metrics['empty_ratio']
            stats['CurvCompact/mean_spread_selected'] = compact_metrics['mean_spread_selected']
            stats['CurvCompact/mean_spread_all'] = compact_metrics['mean_spread_all']
            stats['CurvCompact/mean_rtg_selected'] = compact_metrics['mean_rtg_selected']
            stats['CurvCompact/mean_q_selected'] = compact_metrics['mean_q_selected']
            stats['CurvCompact/mean_centroid_min_dist'] = compact_metrics['mean_centroid_min_dist']
            stats['CurvCompact/selection_success_ratio'] = compact_metrics['selection_success_ratio']
            stats['CurvCompact/mean_dynamic_tau'] = compact_metrics['mean_dynamic_tau']
            stats['CurvCompact/mean_width_safe'] = compact_metrics['mean_width_safe']
            stats['CurvCompact/mean_height_safe'] = compact_metrics['mean_height_safe']

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class KYSActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats


class DiMPSimpleActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'],
                                            train_label=data['train_label'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_test_init_clf = 0
        loss_test_iter_clf = 0

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Loss for the initial filter iteration
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class TargetCandiateMatchingActor(BaseActor):
    """Actor for training the KeepTrack network."""
    def __init__(self, net, objective):
        super().__init__(net, objective)

    def __call__(self, data):
        """
        args:
            data - The input data.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        preds = self.net(**data)

        # Classification losses for the different optimization iterations
        losses = self.objective['target_candidate_matching'](**data, **preds)


        # Total loss
        loss = losses['total'].mean()

        # Log stats
        stats = {
            'Loss/total': loss.item(),
            'Loss/nll_pos': losses['nll_pos'].mean().item(),
            'Loss/nll_neg': losses['nll_neg'].mean().item(),
            'Loss/num_matchable': losses['num_matchable'].mean().item(),
            'Loss/num_unmatchable': losses['num_unmatchable'].mean().item(),
            'Loss/sinkhorn_norm': losses['sinkhorn_norm'].mean().item(),
            'Loss/bin_score': losses['bin_score'].item(),
        }

        if hasattr(self.objective['target_candidate_matching'], 'metrics'):
            metrics = self.objective['target_candidate_matching'].metrics(**data, **preds)

            for key, val in metrics.items():
                stats[key] = torch.mean(val[~torch.isnan(val)]).item()

        return loss, stats


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou']*loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf']*clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious>0].std().item()

        return loss, stats


class TaMOsActor(BaseActor):
    """Actor for training the TaMOs network."""

    def __init__(self, net, objective, loss_weight=None, prob=False, plot_save=None, fg_cls_loss=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.prob = prob
        self.plot_save = plot_save
        self.fg_cls_loss = fg_cls_loss

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def compute_cls_loss(self, test_label, target_scores, test_anno, test_sample_region):
        test_label = test_label.flatten(1, 2)
        target_scores = target_scores.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)

        if self.fg_cls_loss:
            fg_mask = torch.sum(test_sample_region[0], dim=(1, 2)) > 0
            target_scores = target_scores[:, fg_mask]
            test_label = test_label[:, fg_mask]

        clf_loss_test = self.objective['test_clf'](target_scores, test_label, test_anno)
        return clf_loss_test

    def compute_bbreg_loss(self, test_ltrb_target, test_sample_region, bbox_preds):
        test_ltrb_target = test_ltrb_target.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)
        bbox_preds = bbox_preds.flatten(1, 2)
        loss_giou, ious = self.objective['giou'](bbox_preds, test_ltrb_target, test_sample_region)
        return loss_giou, ious

    def compute_iou_pred(self, target_scores, bbox_preds, test_ltrb_target, test_sample_region):
        target_scores = target_scores.flatten(1, 2)
        test_ltrb_target = test_ltrb_target.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)
        bbox_preds = bbox_preds.flatten(1, 2)
        fg_mask = torch.sum(test_sample_region[0], dim=(1, 2)) > 0
        return self.compute_iou_at_max_score_pos(target_scores[:, fg_mask], test_ltrb_target[:, fg_mask],
                                                 bbox_preds[:, fg_mask])

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'],
                                             test_label=data['test_label'],
                                             epoch=data['epoch'])

        stats = {}

        clf_loss_test, loss_giou = 0, 0

        if 'trafo' in target_scores:
            clf_loss_test_trafo = self.compute_cls_loss(data['test_label'], target_scores['trafo'], data['test_anno'],
                                                        data['test_sample_region'])
            clf_loss_test += clf_loss_test_trafo

        if 'lowres' in target_scores:
            clf_loss_test_lowres = self.compute_cls_loss(data['test_label'], target_scores['lowres'], data['test_anno'],
                                                         data['test_sample_region'])
            clf_loss_test += clf_loss_test_lowres

        if 'highres' in target_scores:
            clf_loss_test_highres = self.compute_cls_loss(data['test_label_highres'], target_scores['highres'],
                                                          data['test_anno'], data['test_sample_region_highres'])
            clf_loss_test += clf_loss_test_highres

        if 'trafo' in bbox_preds:
            loss_giou_trafo, ious_trafo = self.compute_bbreg_loss(data['test_ltrb_target'], data['test_sample_region'],
                                                                  bbox_preds['trafo'])
            stats['mIoU_trafo'] = ious_trafo.mean().item()
            loss_giou += loss_giou_trafo

        if 'lowres' in bbox_preds:
            loss_giou_lowres, ious_lowres = self.compute_bbreg_loss(data['test_ltrb_target'],
                                                                    data['test_sample_region'], bbox_preds['lowres'])
            stats['mIoU_lowres'] = ious_lowres.mean().item()
            loss_giou += loss_giou_lowres

        if 'highres' in bbox_preds:
            loss_giou_highres, ious_highres = self.compute_bbreg_loss(data['test_ltrb_target_highres'],
                                                                      data['test_sample_region_highres'],
                                                                      bbox_preds['highres'])
            stats['mIoU_highres'] = ious_highres.mean().item()
            stats['Loss/weighted_GIoU_highres'] = self.loss_weight['giou'] * loss_giou_highres.item()
            loss_giou += loss_giou_highres

        if 'trafo' in target_scores and 'trafo' in bbox_preds:
            ious_pred_center_trafo = self.compute_iou_pred(target_scores['trafo'], bbox_preds['trafo'],
                                                           data['test_ltrb_target'], data['test_sample_region'])
            stats['mIoU_pred_center_trafo'] = ious_pred_center_trafo.mean().item()

        if 'lowres' in target_scores and 'lowres' in bbox_preds:
            ious_pred_center_lowres = self.compute_iou_pred(target_scores['lowres'], bbox_preds['lowres'],
                                                            data['test_ltrb_target'], data['test_sample_region'])
            stats['mIoU_pred_center_lowres'] = ious_pred_center_lowres.mean().item()

        if 'highres' in target_scores and 'highres' in bbox_preds:
            ious_pred_center_highres = self.compute_iou_pred(target_scores['highres'], bbox_preds['highres'],
                                                             data['test_ltrb_target_highres'],
                                                             data['test_sample_region_highres'])
            stats['mIoU_pred_center_highres'] = ious_pred_center_highres.mean().item()

        if 'highres' not in target_scores and 'highres' in bbox_preds:
            if 'trafo' in target_scores:
                target_scores_trafo = target_scores['trafo'].flatten(1, 2)
                target_scores_trafo_interp = F.interpolate(target_scores_trafo,
                                                           data['test_ltrb_target_highres'].shape[-2:], mode='bicubic')

                ious_pred_center_highres = self.compute_iou_pred(
                    target_scores_trafo_interp.reshape(data['test_sample_region_highres'].shape), bbox_preds['highres'],
                    data['test_ltrb_target_highres'], data['test_sample_region_highres'])
                stats['mIoU_pred_center_trafo_highres'] = ious_pred_center_highres.mean().item()

            if 'lowres' in target_scores:
                target_scores_lowres = target_scores['lowres'].flatten(1, 2)
                target_scores_lowres_interp = F.interpolate(target_scores_lowres,
                                                            data['test_ltrb_target_highres'].shape[-2:], mode='bicubic')

                ious_pred_center_highres = self.compute_iou_pred(
                    target_scores_lowres_interp.reshape(data['test_sample_region_highres'].shape),
                    bbox_preds['highres'], data['test_ltrb_target_highres'], data['test_sample_region_highres'])
                stats['mIoU_pred_center_lowres_highres'] = ious_pred_center_highres.mean().item()

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        stats.update({
            'Loss/total': loss.item(),
            'Loss/GIoU': loss_giou.item(),
            'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
            'Loss/clf_loss_test': clf_loss_test.item(),
            'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),

        })

        return loss, stats

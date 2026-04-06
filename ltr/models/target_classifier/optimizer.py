# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.18 (default, Sep 11 2023, 13:40:15) 
# [GCC 11.2.0]
# Embedded file name: /data/lyx/project/pytracking-master/pytracking/../ltr/models/target_classifier/optimizer.py
# Compiled at: 2026-04-03 10:25:42
# Size of source mod 2**32: 54550 bytes
import torch.nn as nn
import torch
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
import math

class DiMPSteepestDescentGN(nn.Module):
    __doc__ = "Optimizer module for DiMP.\n    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.\n    Moreover it learns parameters in the loss itself, as described in the DiMP paper.\n    args:\n        num_iter:  Number of default optimization iterations.\n        feat_stride:  The stride of the input feature.\n        init_step_length:  Initial scaling of the step length (which is then learned).\n        init_filter_reg:  Initial filter regularization weight (which is then learned).\n        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.\n        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.\n        bin_displacement:  The displacement of the bins (level of discritization).\n        mask_init_factor:  Parameter controlling the initialization of the target mask.\n        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.\n        act_param:  Parameter for the score_act.\n        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).\n        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').\n        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.\n        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.\n    "

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, init_filter_reg=0.01, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0, score_act="relu", act_param=None, min_filter_reg=0.001, mask_act="sigmoid", detach_length=float("Inf"), alpha_eps=0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        d = torch.arange(num_dist_bins, dtype=(torch.float32)).reshape(1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[(0, 0, 0, 0)] = 1
        else:
            init_gauss = torch.exp(-0.5 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [
         nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == "sigmoid":
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        else:
            if mask_act == "linear":
                init_bias = 0.5
            else:
                raise ValueError("Unknown activation")
        self.target_mask_predictor = (nn.Sequential)(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == "bentpar":
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        else:
            if score_act == "relu":
                self.score_activation = activation.LeakyReluPar()
                self.score_activation_deriv = activation.LeakyReluParDeriv()
            else:
                raise ValueError("Unknown score activation")

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=(self.min_filter_reg ** 2))
        dmap_offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1, )) - dmap_offset
        dist_map = self.distance_map(center, output_sz)
        label_map = (self.label_map_predictor(dist_map).reshape)(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = (self.target_mask_predictor(dist_map).reshape)(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = (self.spatial_weight_predictor(dist_map).reshape)(num_images, num_sequences, *dist_map.shape[-2:])
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1) * spatial_weight

        backprop_through_learning = self.detach_length > 0
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if not backprop_through_learning or i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            if compute_losses:
                losses.append(((residuals ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
            residuals_mapped = score_mask * (sample_weight * residuals)
            data_grad = filter_layer.apply_feat_transpose(feat,
              residuals_mapped, filter_sz, training=(self.training))
            weights_grad = data_grad + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,
                                                                                                     2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = self.score_activation(scores, target_mask)
            losses.append((((sample_weight * (scores - label_map)) ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
        return (weights, weight_iterates, losses)


class DiMPL2SteepestDescentGN(nn.Module):
    __doc__ = "A simpler optimizer module that uses L2 loss.\n    args:\n        num_iter:  Number of default optimization iterations.\n        feat_stride:  The stride of the input feature.\n        init_step_length:  Initial scaling of the step length (which is then learned).\n        gauss_sigma:  The standard deviation of the label function.\n        hinge_threshold:  Threshold for the hinge-based loss (see DiMP paper).\n        init_filter_reg:  Initial filter regularization weight (which is then learned).\n        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).\n        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.\n        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.\n    "

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, gauss_sigma=1.0, hinge_threshold=-999, init_filter_reg=0.01, min_filter_reg=0.001, detach_length=float("Inf"), alpha_eps=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.hinge_threshold = hinge_threshold
        self.gauss_sigma = gauss_sigma
        self.alpha_eps = alpha_eps

    def get_label(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange((output_sz[0]), dtype=(torch.float32)).reshape(1, 1, -1, 1).to(center.device)
        k1 = torch.arange((output_sz[1]), dtype=(torch.float32)).reshape(1, 1, 1, -1).to(center.device)
        g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k0 - center[:, :, 0].reshape(*center.shape[:2], 1, 1)) ** 2)
        g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k1 - center[:, :, 1].reshape(*center.shape[:2], 1, 1)) ** 2)
        gauss = g0 * g1
        return gauss

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=(self.min_filter_reg ** 2))
        dmap_offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1, )) - dmap_offset
        label_map = self.get_label(center, output_sz)
        target_mask = (label_map > self.hinge_threshold).float()
        label_map *= target_mask
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        else:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1)
            weight_iterates = [weights]
            losses = []
            for i in range(num_iter):
                if i > 0:
                    if i % self.detach_length == 0:
                        weights = weights.detach()
                scores = filter_layer.apply_filter(feat, weights)
                scores_act = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
                score_mask = target_mask + (1.0 - target_mask) * (scores.detach() > 0).float()
                residuals = sample_weight * (scores_act - label_map)
                if compute_losses:
                    losses.append(((residuals ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
                residuals_mapped = score_mask * (sample_weight * residuals)
                data_grad = filter_layer.apply_feat_transpose(feat,
                  residuals_mapped, filter_sz, training=(self.training))
                weights_grad = data_grad + reg_weight * weights
                scores_grad = filter_layer.apply_filter(feat, weights_grad)
                scores_grad = sample_weight * (score_mask * scores_grad)
                alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
                alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,
                                                                                                         2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
                alpha = alpha_num / alpha_den
                weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
                weight_iterates.append(weights)
            else:
                if compute_losses:
                    scores = filter_layer.apply_filter(feat, weights)
                    scores = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
                    losses.append((((sample_weight * (scores - label_map)) ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
                return (weights, weight_iterates, losses)


class PrDiMPSteepestDescentNewton(nn.Module):
    __doc__ = "Optimizer module for PrDiMP.\n    It unrolls the steepest descent with Newton iterations to optimize the target filter. See the PrDiMP paper.\n    args:\n        num_iter:  Number of default optimization iterations.\n        feat_stride:  The stride of the input feature.\n        init_step_length:  Initial scaling of the step length (which is then learned).\n        init_filter_reg:  Initial filter regularization weight (which is then learned).\n        gauss_sigma:  The standard deviation to use for the label density function.\n        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).\n        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.\n        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.\n        init_uni_weight:  Weight of uniform label distribution.\n        normalize_label:  Wheter to normalize the label distribution.\n        label_shrink:  How much to shrink to label distribution.\n        softmax_reg:  Regularization in the denominator of the SoftMax.\n        label_threshold:  Threshold probabilities smaller than this.\n    "

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, init_filter_reg=0.01, gauss_sigma=1.0, min_filter_reg=0.001, detach_length=float("Inf"), alpha_eps=0.0, init_uni_weight=None, normalize_label=False, label_shrink=0, softmax_reg=None, label_threshold=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.uni_weight = 0 if init_uni_weight is None else init_uni_weight
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold

    def get_label_density(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange((output_sz[0]), dtype=(torch.float32)).reshape(1, 1, -1, 1).to(center.device)
        k1 = torch.arange((output_sz[1]), dtype=(torch.float32)).reshape(1, 1, 1, -1).to(center.device)
        dist0 = (k0 - center[:, :, 0].reshape(*center.shape[:2], 1, 1)) ** 2
        dist1 = (k1 - center[:, :, 1].reshape(*center.shape[:2], 1, 1)) ** 2
        if self.gauss_sigma == 0:
            dist0_view = dist0.reshape(-1, dist0.shape[-2])
            dist1_view = dist1.reshape(-1, dist1.shape[-1])
            one_hot0 = torch.zeros_like(dist0_view)
            one_hot1 = torch.zeros_like(dist1_view)
            one_hot0[(torch.arange(one_hot0.shape[0]), dist0_view.argmin(dim=(-1)))] = 1.0
            one_hot1[(torch.arange(one_hot1.shape[0]), dist1_view.argmin(dim=(-1)))] = 1.0
            gauss = one_hot0.reshape(dist0.shape) * one_hot1.reshape(dist1.shape)
        else:
            g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist0)
            g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist1)
            gauss = g0 / (2 * math.pi * self.gauss_sigma ** 2) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= gauss.sum(dim=(-2, -1), keepdim=True) + 1e-08
        label_dens = (1.0 - self.label_shrink) * ((1.0 - self.uni_weight) * gauss + self.uni_weight / (output_sz[0] * output_sz[1]))
        return label_dens

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=(self.min_filter_reg ** 2))
        offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1, )) - offset
        label_density = self.get_label_density(center, output_sz)
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images]).to(feat.device)
        else:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.reshape(num_images, num_sequences, 1, 1)
            exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

            def _compute_loss(scores, weights):
                return torch.sum(sample_weight.reshape(sample_weight.shape[0], -1) * (torch.log(scores.exp().sum(dim=(-2,
                                                                                                                      -1)) + exp_reg) - (label_density * scores).sum(dim=(-2,
                                                                                                                                                                          -1)))) / num_sequences + reg_weight * (weights ** 2).sum() / num_sequences

            weight_iterates = [
             weights]
            losses = []
            for i in range(num_iter):
                if i > 0:
                    if i % self.detach_length == 0:
                        weights = weights.detach()
                scores = filter_layer.apply_filter(feat, weights)
                scores_softmax = activation.softmax_reg((scores.reshape(num_images, num_sequences, -1)), dim=2, reg=(self.softmax_reg)).reshape(scores.shape)
                residuals = sample_weight * (scores_softmax - label_density)
                score_mask = (sample_weight * scores_softmax * (1.0 - scores_softmax)).reshape(scores.shape)
                if compute_losses:
                    losses.append(((residuals ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
                residuals_mapped = score_mask * (sample_weight * residuals)
                data_grad = filter_layer.apply_feat_transpose(feat,
                  residuals_mapped, filter_sz, training=(self.training))
                weights_grad = data_grad + reg_weight * weights
                scores_grad = filter_layer.apply_filter(feat, weights_grad)
                scores_grad = sample_weight * (score_mask * scores_grad)
                alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
                alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,
                                                                                                         2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
                alpha = alpha_num / alpha_den
                weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
                weight_iterates.append(weights)
            else:
                if compute_losses:
                    scores = filter_layer.apply_filter(feat, weights)
                    losses.append(_compute_loss(scores, weights))
                return (weights, weight_iterates, losses)


class CurvatureAwareDiMP(nn.Module):
    __doc__ = 'Curvature-Aware Anisotropic Regularization for DiMP.\n\n    Replaces isotropic L2 regularization ½λ||w||² with anisotropic\n    curvature-aware regularization ½λ(w - w*)ᵀdiag(H̄)(w - w*), where H̄\n    is an EMA of the empirical quasi-Hessian diagonal accumulated from\n    the memory bank. High-curvature directions (large H̄ entries) are\n    anchored strongly toward the anchor, while low-curvature directions\n    receive weaker regularization and can adapt more freely.\n\n    Key implementation insight: diag(JᵀWJ) never requires instantiating\n    the Jacobian matrix. Because DiMP\'s score map is computed as a\n    per-channel spatial convolution, JᵀWJ is block-diagonal across\n    channels. The diagonal element for channel c is simply the\n    sample/mask-weighted sum of squared features in that channel:\n\n        diag(H)ᵗ = Σ_{h,w} W_{h,w} · score_mask_{h,w} · (x_{c,h,w})²\n\n    This is an element-wise weighted sum per channel — O(D) memory\n    and O(HWD) time, completely independent of the filter kernel size.\n\n    Key theoretical claim (distilled from extensive discussion):\n      Rather than claiming "FIM protects small targets" (which is false,\n      since small targets have low gradient energy and hence low H̄),\n      the core contribution is: replacing the isotropic scalar damping\n      λ with the anisotropic tensor λ·diag(H̄) yields better background\n      suppression stability in non-i.i.d. online data streams.  High-H̄\n      directions are directions where the filter must work hard to\n      suppress large-area background — we anchor these so the filter\n      cannot trade away background suppression for short-term clutter\n      fitting.  Low-H̄ directions receive less regularization and are\n      free to adapt to current-frame evidence (including genuine small-\n      target appearance changes); dead channels and noise simply remain\n      inactive.\n    '

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, init_filter_reg=0.01, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0, score_act="relu", act_param=None, min_filter_reg=0.001, mask_act="sigmoid", detach_length=float("Inf"), alpha_eps=0, use_curvature_reg=False, use_curvature_shield=True, ema_momentum=0.01, anchor_update_interval=5, init_anchor_from="init", curvature_reg_weight=1.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.use_curvature_reg = use_curvature_reg
        self.use_curvature_shield = use_curvature_shield
        self.ema_momentum = ema_momentum
        self.anchor_update_interval = anchor_update_interval
        self.init_anchor_from = init_anchor_from
        self.curvature_reg_weight = curvature_reg_weight
        self.H_diag_ema = None
        self.H_diag_norm_ema = None
        self.weights_anchor = None
        self._anchor_frame_counter = 0
        self._anchor_updated_in_frame = False
        self._last_diag_state = {}
        self._frame_scores_for_shield = None
        d = torch.arange(num_dist_bins, dtype=(torch.float32)).reshape(1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[(0, 0, 0, 0)] = 1
        else:
            init_gauss = torch.exp(-0.5 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [
         nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == "sigmoid":
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        else:
            if mask_act == "linear":
                init_bias = 0.5
            else:
                raise ValueError("Unknown activation")
        self.target_mask_predictor = (nn.Sequential)(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == "bentpar":
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        else:
            if score_act == "relu":
                self.score_activation = activation.LeakyReluPar()
                self.score_activation_deriv = activation.LeakyReluParDeriv()
            else:
                raise ValueError("Unknown score activation")

    def _compute_H_diag(self, feat, target_mask, sample_weight, spatial_weight, filter_sz):
        """Compute per-spatial-element diagonal of the empirical quasi-Hessian.

        DiMP forward (cross-correlation with SAME padding = centred shift-sum):
            s[h, w] = Σ_c Σ_{i,j} w[c, i, j] · x[c, h+i-K//2, w+j-K//2]
        Jacobian:  ∂s[h,w]/∂w[c,i,j] = x[c, h+i-K//2, w+j-K//2]

        Hessian diagonal:
            H_diag[c, i, j] = Σ_{h,w} W[h,w] · score_mask[h,w] · x[c, h+i-K//2, w+j-K//2]²

        Implementation: for each image in the batch, compute the weighted feature
        energy (x_sq * eff) and pool it from feature-map resolution (H×W) down to
        filter resolution (K_h×K_w) via adaptive_avg_pool2d.  Average across images.
        This preserves a coarse K_h×K_w spatial map so the filter centre (high
        activation → high H) is distinguished from the edges (low activation → low H).

        RISK-1 FIX: The naive (feat**2 * mask).sum(dim=(-2,-1)) collapses all
        spatial filter positions into a single per-channel scalar, destroying
        spatial anisotropy.  adaptive_avg_pool2d preserves a coarse K_h×K_w spatial
        map so the filter centre (high H) gets a different H from the edges.

        Args:
            feat: 5D (NI, NS, D, H, W) — standard multi-sequence call path.
                  4D (NI, D, H, W) when num_sequences=1 is embedded in the batch
                  (as in DiMP update_classifier).
            target_mask: 4D (NI, NS, H, W) or (NI, H, W) when embedded.
            sample_weight: 1D (NI,) or 2D (NI, NS) raw per-image scalars.
            spatial_weight: 4D (NI, NS, H, W) or (NI, H, W) when embedded.
            filter_sz: (K_h, K_w)

        Returns:
            H_diag: (num_sequences, feat_dim, K_h, K_w)
        """
        if feat.dim() == 4:
            # Embedded single-sequence path: (NI, D, H, W) -> (NI, 1, D, H, W)
            feat = feat.unsqueeze(1)

        num_images = feat.shape[0]
        num_sequences = feat.shape[1]
        K_h, K_w = filter_sz

        # Normalize masks/weights to (NI, NS, 1, H, W)
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        if target_mask.dim() == 4:
            target_mask = target_mask.unsqueeze(2)

        if spatial_weight.dim() == 3:
            spatial_weight = spatial_weight.unsqueeze(1)
        if spatial_weight.dim() == 4:
            spatial_weight = spatial_weight.unsqueeze(2)

        x_sq = feat ** 2

        # sample weight broadcast to (NI, NS, 1, 1, 1)
        if isinstance(sample_weight, torch.Tensor):
            if sample_weight.dim() == 1:
                sw = sample_weight.sqrt().reshape(num_images, 1, 1, 1, 1)
            elif sample_weight.dim() == 2:
                sw = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1, 1)
            else:
                sw = sample_weight.reshape(num_images, num_sequences, 1, 1, 1)
        else:
            sw = math.sqrt(1.0 / num_images)

        eff = sw * spatial_weight * target_mask
        feat_h, feat_w = feat.shape[-2], feat.shape[-1]
        if eff.shape[-2] != feat_h or eff.shape[-1] != feat_w:
            ni, ns = eff.shape[0], eff.shape[1]
            eff = F.adaptive_avg_pool2d(eff.reshape(ni * ns, 1, eff.shape[-2], eff.shape[-1]), (
             feat_h, feat_w)).reshape(ni, ns, 1, feat_h, feat_w)
        weighted_energy = x_sq * eff
        ni, ns, d = weighted_energy.shape[0], weighted_energy.shape[1], weighted_energy.shape[2]
        feat_h_in = weighted_energy.shape[-2]
        feat_w_in = weighted_energy.shape[-1]
        _pool_scale = feat_h_in / K_h * (feat_w_in / K_w)
        H_diag_per_sample = F.adaptive_avg_pool2d(weighted_energy.reshape(ni * ns, d, feat_h_in, feat_w_in), (
         K_h, K_w)).reshape(ni, ns, d, K_h, K_w) * _pool_scale
        H_diag = H_diag_per_sample.mean(dim=0)
        return H_diag

    def _update_slow_anchor(self, weights, feat, target_mask, sample_weight, spatial_weight):
        """Synchronized EMA update of slow anchor weights and quasi-Hessian EMA.

        RISK-3 FIX: w* and H̄ are updated together — same EMA momentum β,
        same gate condition, same frame.  This prevents "temporal tearing":
        a high-H̄ (punishing hard) paired with an old w* (referencing a
        different regime) would apply unreasonable force in the wrong regime.

        [STATE-AWARE] Also updates H_diag_norm_ema (running EMA of per-channel max).
        """
        if not self.use_curvature_reg:
            return
        else:
            self._anchor_updated_in_frame = False
            filter_sz = (
             weights.shape[-2], weights.shape[-1])
            H_diag = self._compute_H_diag(feat, target_mask, sample_weight, spatial_weight, filter_sz)
            if self.H_diag_ema is None:
                self.H_diag_ema = H_diag.detach().clone()
                self.weights_anchor = weights.detach().clone()
                self._anchor_updated_in_frame = True
            else:
                self.H_diag_ema = (1 - self.ema_momentum) * self.H_diag_ema + self.ema_momentum * H_diag.detach()
                anchor_momentum = max(1e-06, min(1.0, self.ema_momentum * 0.2))
                self.weights_anchor = (1 - anchor_momentum) * self.weights_anchor + anchor_momentum * weights.detach()
                self._anchor_updated_in_frame = True

    def _extract_radar_state(self, scores):
        """Extract per-frame radar state used by shield and anchor gating."""
        thresh = getattr(self, "_curv_low_score_th", 0.35)
        bg_thresh = getattr(self, "_curv_bg_thresh", 0.4)
        bg_radius = getattr(self, "_curv_bg_radius", 2)

        max_score_val = None
        dist_from_center_val = None
        is_lost_flag = False
        is_attacked_flag = False

        if scores is None:
            return {
                'max_score': max_score_val,
                'dist_from_center': dist_from_center_val,
                'is_lost': is_lost_flag,
                'is_attacked': is_attacked_flag,
            }

        if scores.dim() == 2:
            scores = scores.unsqueeze(0).unsqueeze(0)
        elif scores.dim() == 3:
            scores = scores.unsqueeze(1)

        if scores.dim() < 4:
            return {
                'max_score': max_score_val,
                'dist_from_center': dist_from_center_val,
                'is_lost': is_lost_flag,
                'is_attacked': is_attacked_flag,
            }

        frame_scores = scores[-1]
        if frame_scores.dim() == 2:
            frame_scores = frame_scores.unsqueeze(0)

        ns, h_map, w_map = frame_scores.shape
        flat_scores = frame_scores.reshape(ns, -1)
        max_score, max_idx = flat_scores.max(dim=-1)

        peak_y = (max_idx // w_map).to(frame_scores.dtype)
        peak_x = (max_idx % w_map).to(frame_scores.dtype)
        center_y = (h_map - 1) / 2.0
        center_x = (w_map - 1) / 2.0
        dist_from_center = torch.sqrt((peak_y - center_y) ** 2 + (peak_x - center_x) ** 2)

        is_lost = max_score < thresh
        is_attacked = (dist_from_center > bg_radius) & (max_score > bg_thresh)

        return {
            'max_score': float(max_score.max().item()),
            'dist_from_center': float(dist_from_center.max().item()),
            'is_lost': bool(is_lost.any().item()),
            'is_attacked': bool(is_attacked.any().item()),
        }

    def _update_H_diag_norm_ema(self, H_diag):
        """Update the running-EMA of per-channel max of H_diag_ema.

        This provides a stable normalization denominator that does NOT
        monotonically decay when target energy naturally drops (e.g. pose change).
        Instead it reflects the long-term peak curvature energy seen so far.
        """
        h_flat = H_diag.reshape(H_diag.shape[0], H_diag.shape[1], -1)
        ch_p95 = torch.quantile(h_flat, 0.95, dim=-1)
        if self.H_diag_norm_ema is None:
            self.H_diag_norm_ema = ch_p95.detach().clone()
        else:
            self.H_diag_norm_ema = (1 - self.ema_momentum) * self.H_diag_norm_ema + self.ema_momentum * ch_p95.detach()

    def _get_curvature_state(self):
        """Compute max-normalized H and state-aware curvature weight.

        Returns (H_norm, H_inv, current_weight) where:
          H_norm: safe max-normalization (values in 0~1) tracking the target.
          H_inv:  1 - H_norm, tracking the background.
          current_weight: scalar curvature weight, default 1.0 (soft) or 10.0 (hard shield).

        Shield logic:
          hard  (cur_weight = curv_hard_weight) when:
            1. scores.max() < curv_low_score_th          — target confidence drops
            2. peripheral_max > curv_bg_thresh           — distractor in background
          soft  (cur_weight = curv_soft_weight) otherwise.
        If use_curvature_shield=False, always returns soft weight.
        """
        soft_w = getattr(self, "_curv_soft_weight", 1.0)
        hard_w = getattr(self, "_curv_hard_weight", 10.0)
        shield_on = getattr(self, "use_curvature_shield", True)
        scores = getattr(self, "_frame_scores_for_shield", None)
        cur_weight = float(soft_w)
        radar = self._extract_radar_state(scores) if shield_on else {
            'max_score': None,
            'dist_from_center': None,
            'is_lost': False,
            'is_attacked': False,
        }
        max_score_val = radar['max_score']
        dist_from_center_val = radar['dist_from_center']
        is_lost_flag = radar['is_lost']
        is_attacked_flag = radar['is_attacked']

        if shield_on:
            use_soft_gating = getattr(self, "_use_curv_soft_gating", True)
            if use_soft_gating and max_score_val is not None and dist_from_center_val is not None:
                import math
                thresh = getattr(self, "_curv_low_score_th", 0.35)
                bg_thresh = getattr(self, "_curv_bg_thresh", 0.4)
                bg_radius = getattr(self, "_curv_bg_radius", 2)
                
                # Soften the step function using Sigmoid (temperature=0.03 means ~0.1 score difference is a full transition)
                # Gate 1: Lost (High punishment when score < thresh)
                w_lost = 1.0 / (1.0 + math.exp((max_score_val - thresh) / 0.03))
                
                # Gate 2: Attacked (High punishment when dist > bg_radius and score > bg_thresh)
                w_bg_score = 1.0 / (1.0 + math.exp((bg_thresh - max_score_val) / 0.03))
                w_bg_dist = 1.0 if dist_from_center_val > bg_radius else 0.0
                w_attack = w_bg_score * w_bg_dist
                
                # Interpolate cur_weight between soft_w and hard_w
                gate_val = max(w_lost, w_attack)
                cur_weight = float(soft_w + gate_val * (hard_w - soft_w))
            else:
                # Original Hard threshold logic
                if is_lost_flag or is_attacked_flag:
                    cur_weight = float(hard_w)

        if self.H_diag_ema is not None:
            h_flat = self.H_diag_ema.reshape(self.H_diag_ema.shape[0], self.H_diag_ema.shape[1], -1)
            h_p95 = torch.quantile(h_flat, 0.95, dim=-1)
            H_norm = self.H_diag_ema / (h_p95.unsqueeze(-1).unsqueeze(-1) + 1e-08)
            H_norm = torch.clamp(H_norm, 0.0, 1.0)
            H_inv = 1.0 - H_norm
            h_scale = float(h_p95.mean().item())
        else:
            H_norm = self.H_diag_ema
            H_inv = None
            h_scale = None

        if max_score_val is None and getattr(self, "_frame_max_score", None) is not None:
            max_score_val = float(self._frame_max_score)

        self._last_diag_state = {
            'max_score': max_score_val,
            'dist_from_center': dist_from_center_val,
            'is_attacked': bool(is_attacked_flag),
            'is_lost': bool(is_lost_flag),
            'anchor_updated': bool(self._anchor_updated_in_frame),
            'H_p95_mean': h_scale,
            'cur_weight': float(cur_weight),
        }
        return (H_norm, H_inv, cur_weight)

    def reset_anchor(self):
        """Reset slow anchor state. Call when initialising a new tracking sequence."""
        self.H_diag_ema = None
        self.H_diag_norm_ema = None
        self.weights_anchor = None
        self._anchor_frame_counter = 0
        self._anchor_updated_in_frame = False
        self._last_diag_state = {}

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True, update_anchor=True, scores=None):
        """Optimise the target filter with optional curvature-aware regularization.

        All changes relative to baseline DiMPSteepestDescentGN are marked
        [CURVATURE-AWARE].

        Args:
            update_anchor: Whether to update slow anchors from this batch.
                          Set False during validation / inference on a frozen model.
            scores: 4D (NI, NS, H, W) raw classifier score map from the current frame,
                    used for state-aware shield decision (low max_score → hard mode).
        """
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (
            feat.shape[-2] + (weights.shape[-2] + 1) % 2,
            feat.shape[-1] + (weights.shape[-1] + 1) % 2,
        )

        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg ** 2)

        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = (
            ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1,))
            - dmap_offset
        )
        dist_map = self.distance_map(center, output_sz)

        label_map = self.label_map_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )
        # 官方原生软掩码（留给底层的残差预测使用，完美匹配官方预训练参数）
        target_mask = self.target_mask_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )

        # 曲率定制版硬切掩码（专供 _compute_H_diag 分离背景用）
        h_feat, w_feat = output_sz
        device = bb.device
        bb_w = bb[..., 2].reshape(num_images, num_sequences, 1, 1)
        bb_h = bb[..., 3].reshape(num_images, num_sequences, 1, 1)
        grid_w = torch.clamp(bb_w / self.feat_stride, min=1.0)
        grid_h = torch.clamp(bb_h / self.feat_stride, min=1.0)
        cy = center[:, 0].reshape(num_images, num_sequences, 1, 1)
        cx = center[:, 1].reshape(num_images, num_sequences, 1, 1)
        y_grid = torch.arange(h_feat, device=device).view(1, 1, h_feat, 1)
        x_grid = torch.arange(w_feat, device=device).view(1, 1, 1, w_feat)
        in_h = (y_grid - cy).abs() <= (grid_h / 2.0)
        in_w = (x_grid - cx).abs() <= (grid_w / 2.0)
        curv_target_mask = (in_h & in_w).float()

        self._last_target_mask = curv_target_mask.detach()
        raw_sw = sample_weight
        self._frame_max_score = scores.max().item() if scores is not None else None
        self._frame_scores_for_shield = scores.detach().clone() if scores is not None else None
        radar_now = self._extract_radar_state(self._frame_scores_for_shield)
        self._anchor_updated_in_frame = False
        self._last_diag_state = {
            'max_score': radar_now['max_score'] if radar_now['max_score'] is not None else (float(self._frame_max_score) if self._frame_max_score is not None else None),
            'dist_from_center': radar_now['dist_from_center'],
            'is_attacked': bool(radar_now['is_attacked']),
            'is_lost': bool(radar_now['is_lost']),
            'anchor_updated': False,
            'H_p95_mean': float(self.H_diag_norm_ema.mean().item()) if self.H_diag_norm_ema is not None else None,
            'cur_weight': float(getattr(self, '_curv_soft_weight', 1.0)),
        }

        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1) * spatial_weight

        if update_anchor and self.use_curvature_reg:
            raw_H = self._compute_H_diag(feat, curv_target_mask, raw_sw, spatial_weight, filter_sz)
            self._last_H_diag_raw = raw_H
            self._update_slow_anchor(weights, feat, curv_target_mask, raw_sw, spatial_weight)

        backprop_through_learning = self.detach_length > 0
        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if not backprop_through_learning or (i > 0 and i % self.detach_length == 0):
                weights = weights.detach()

            scores_iter = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores_iter, target_mask)
            score_mask = self.score_activation_deriv(scores_iter, target_mask)
            residuals = sample_weight * (scores_act - label_map)

            if self.use_curvature_reg and self.weights_anchor is not None:
                H_norm, H_inv, cur_weight = self._get_curvature_state()
                reg_diff_anchor = weights - self.weights_anchor
                reg_diff_zero = weights
                curv_reg = 0.5 * cur_weight * reg_weight * (
                    (H_norm * (reg_diff_anchor ** 2)) +
                    (H_inv * (reg_diff_zero ** 2))
                ).sum()
            else:
                H_norm = None
                cur_weight = 1.0
                curv_reg = 0.0

            if compute_losses:
                if self.use_curvature_reg and self.weights_anchor is not None:
                    reg_energy_loss = curv_reg * num_sequences
                else:
                    reg_energy_loss = reg_weight * (weights ** 2).sum()
                losses.append(((residuals ** 2).sum() + reg_energy_loss) / num_sequences)

            if self.use_curvature_reg:
                residuals_mapped = score_mask * (sample_weight * residuals)
            else:
                residuals_mapped = score_mask * (sample_weight * residuals)
            
            data_grad = filter_layer.apply_feat_transpose(
                feat, residuals_mapped, filter_sz, training=self.training
            )

            if self.use_curvature_reg and self.weights_anchor is not None:
                curv_grad = cur_weight * reg_weight * (
                    (H_norm * reg_diff_anchor) +
                    (H_inv * reg_diff_zero)
                )
                weights_grad = data_grad + curv_grad
            else:
                weights_grad = data_grad + reg_weight * weights

            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))

            if self.use_curvature_reg and self.weights_anchor is not None:
                curv_denom = cur_weight * reg_weight * alpha_num
                alpha_den = (
                    (scores_grad * scores_grad)
                    .reshape(num_images, num_sequences, -1)
                    .sum(dim=(0, 2))
                    + self.alpha_eps * alpha_num
                    + curv_denom
                ).clamp(1e-8)
            else:
                alpha_den = (
                    (scores_grad * scores_grad)
                    .reshape(num_images, num_sequences, -1)
                    .sum(dim=(0, 2))
                    + (reg_weight + self.alpha_eps) * alpha_num
                ).clamp(1e-8)

            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
            weight_iterates.append(weights)

        if compute_losses:
            scores_final = filter_layer.apply_filter(feat, weights)
            scores_final = self.score_activation(scores_final, target_mask)
            if self.use_curvature_reg and self.weights_anchor is not None:
                H_norm, H_inv, cur_weight = self._get_curvature_state()
                reg_diff_anchor = weights - self.weights_anchor
                reg_diff_zero = weights
                curv_reg = 0.5 * cur_weight * reg_weight * (
                    (H_norm * (reg_diff_anchor ** 2)) +
                    (H_inv * (reg_diff_zero ** 2))
                ).sum()
                reg_energy_loss = curv_reg * num_sequences
            else:
                reg_energy_loss = reg_weight * (weights ** 2).sum()

            losses.append((((sample_weight * (scores_final - label_map)) ** 2).sum() + reg_energy_loss) / num_sequences)

        return weights, weight_iterates, losses

# okay decompiling /data/lyx/project/pytracking-master/ltr/models/target_classifier/__pycache__/optimizer.cpython-38.pyc


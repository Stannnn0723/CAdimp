import re

path = '/data/lyx/project/pytracking-master/ltr/models/target_classifier/optimizer.py'
with open(path, 'r') as f:
    text = f.read()

# Pattern 1 (in DiMPSteepestDescentGN)
old_1 = """        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])"""

new_1 = """        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])

        # --- DYNAMIC TARGET MASK ---
        # Instead of a hardcoded ~5x5 region, dynamically map the actual BBox W/H
        # scaled by feat_stride onto the feature grid.
        H_feat, W_feat = output_sz
        device = bb.device
        
        bb_w = bb[..., 2].reshape(num_images, num_sequences, 1, 1)
        bb_h = bb[..., 3].reshape(num_images, num_sequences, 1, 1)
        grid_w = torch.clamp((bb_w / self.feat_stride), min=1.0)
        grid_h = torch.clamp((bb_h / self.feat_stride), min=1.0)
        
        cy = center[:, 0].reshape(num_images, num_sequences, 1, 1)
        cx = center[:, 1].reshape(num_images, num_sequences, 1, 1)
        
        y_grid = torch.arange(H_feat, device=device).view(1, 1, H_feat, 1)
        x_grid = torch.arange(W_feat, device=device).view(1, 1, 1, W_feat)
        
        in_h = (y_grid - cy).abs() <= (grid_h / 2.0)
        in_w = (x_grid - cx).abs() <= (grid_w / 2.0)
        target_mask = (in_h & in_w).float()"""

# Pattern 2 (in CurvatureAwareDiMP)
old_2 = """        label_map = self.label_map_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )
        target_mask = self.target_mask_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )"""

new_2 = """        label_map = self.label_map_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:]
        )

        # --- DYNAMIC TARGET MASK ---
        # Replace hardcoded mask logic with dynamic grid scaling based on true target W/H
        H_feat, W_feat = output_sz
        device = bb.device
        
        bb_w = bb[..., 2].reshape(num_images, num_sequences, 1, 1)
        bb_h = bb[..., 3].reshape(num_images, num_sequences, 1, 1)
        grid_w = torch.clamp((bb_w / self.feat_stride), min=1.0)
        grid_h = torch.clamp((bb_h / self.feat_stride), min=1.0)
        
        cy = center[:, 0].reshape(num_images, num_sequences, 1, 1)
        cx = center[:, 1].reshape(num_images, num_sequences, 1, 1)
        
        y_grid = torch.arange(H_feat, device=device).view(1, 1, H_feat, 1)
        x_grid = torch.arange(W_feat, device=device).view(1, 1, 1, W_feat)
        
        in_h = (y_grid - cy).abs() <= (grid_h / 2.0)
        in_w = (x_grid - cx).abs() <= (grid_w / 2.0)
        target_mask = (in_h & in_w).float()"""

text = text.replace(old_1, new_1)
text = text.replace(old_2, new_2)

with open(path, 'w') as f:
    f.write(text)
print("Patch applied")

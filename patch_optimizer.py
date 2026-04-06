with open("/data/lyx/project/pytracking-master/ltr/models/target_classifier/optimizer.py", "r") as f:
    text = f.read()

import re
old_text = r"""            shield_active = low_confidence | distractor_nearby
            cur_weight = torch.where(
                shield_active.unsqueeze(-1).unsqueeze(-1).expand_as(self.H_diag_ema\[:1\]),
                hard_w \* torch.ones_like\(self.H_diag_ema\[:1\]\),
                soft_w \* torch.ones_like\(self.H_diag_ema\[:1\]\)
            )\[0, :, 0, 0\].mean\(\)\s*# scalar"""

new_text = """            shield_active = low_confidence | distractor_nearby
            cur_weight = torch.where(
                shield_active,
                torch.tensor(hard_w, device=shield_active.device),
                torch.tensor(soft_w, device=shield_active.device)
            ).view(-1, 1, 1, 1).float()"""

text = re.sub(old_text, new_text, text, flags=re.MULTILINE)

with open("/data/lyx/project/pytracking-master/ltr/models/target_classifier/optimizer.py", "w") as f:
    f.write(text)

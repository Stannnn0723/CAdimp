with open("/data/lyx/project/pytracking-master/ltr/models/target_classifier/optimizer.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "shield_active = low_confidence | distractor_nearby" in line:
        if "view(-1, 1, 1, 1)" in line or "| distractor_nearby" in line:
            pass # We will re-insert clean version
        new_lines.append("            shield_active = low_confidence | distractor_nearby\n")
        new_lines.append("            cur_weight = torch.where(\n")
        new_lines.append("                shield_active,\n")
        new_lines.append("                torch.tensor(hard_w, device=shield_active.device),\n")
        new_lines.append("                torch.tensor(soft_w, device=shield_active.device)\n")
        new_lines.append("            ).view(-1, 1, 1, 1).float()\n")
        skip = True
        continue
    if skip:
        if "else:" in line and "cur_weight = soft_w" in lines[lines.index(line)+1]:
            skip = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open("/data/lyx/project/pytracking-master/ltr/models/target_classifier/optimizer.py", "w") as f:
    f.writelines(new_lines)

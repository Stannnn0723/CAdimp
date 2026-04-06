import os
dimp_path = '/data/lyx/project/pytracking-master/pytracking/tracker/dimp/dimp.py'
with open(dimp_path, 'r') as f:
    text = f.read()

new_text = text.replace(
    "scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)",
    "try:\n                scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)\n            except Exception as e:\n                print('REG_VAL SHAPE:', getattr(reg_val, 'shape', type(reg_val)))\n                print('SCORES VIEW SHAPE:', scores_view.shape)\n                raise e\n            if scores_softmax.numel() != scores.numel():\n                print('WEIRD SCORE SHAPE:', scores_softmax.shape, 'vs original', scores.shape)\n                print('REG_VAL SHAPE:', getattr(reg_val, 'shape', type(reg_val)))"
)

with open(dimp_path, 'w') as f:
    f.write(new_text)


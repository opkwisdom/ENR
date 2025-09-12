import pickle
import torch
from torch.utils.data import Dataset, random_split
import os


in_path = "../../data/bge_base_en_v1.5/pretrain.pickle"
out_dir = "../../data/msmarco/pretrain"
val_ratio = 0.01
seed = 42

with open(in_path, 'rb') as f:
    full_list = pickle.load(f)

val_ratio = 0.01
n_total = len(full_list)
n_val = int(n_total * val_ratio)


# 재현 가능한 셔플 인덱스
g = torch.Generator().manual_seed(seed)
perm = torch.randperm(n_total, generator=g)
val_idx   = perm[:n_val].tolist()
train_idx = perm[n_val:].tolist()

# 서브리스트 생성 (원본 복사 아님: 참조만 담는 리스트 → 메모리 부담 적음)
train_list = [full_list[i] for i in train_idx]
dev_list   = [full_list[i] for i in val_idx]

# 저장
train_path = os.path.join(out_dir, "pretrain_train.pickle")
dev_path   = os.path.join(out_dir, "pretrain_dev.pickle")

with open(train_path, "wb") as f:
    pickle.dump(train_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(dev_path, "wb") as f:
    pickle.dump(dev_list, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved: {train_path} (Size: {len(train_list)})")
print(f"Saved: {dev_path}   (Size: {len(dev_list)})")
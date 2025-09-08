import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_distance_over_codebook(codebook):
    L, K, D = codebook.shape
    distance_map = torch.zeros((L, K, K), dtype=codebook.dtype, device=codebook.device)
    
    for i in range(L):
        c = codebook[i]
        diff = c.unsqueeze(1) - c.unsqueeze(0)  # (K, K, D), broadcasting
        dist = (diff ** 2).sum(dim=-1)
        distance_map[i] = dist
    return distance_map


def plot_distance_matrix(distance_matrix, title="Distance_0", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, cmap="viridis", square=True,
                vmin=0, vmax=3)
    plt.title(title)
    plt.xlabel("Centroid Index")
    plt.ylabel("Centroid Index")
    
    filename = title + ".png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    # codebook = torch.load("../../data/bge_base_en_v1.5/codebook_normed.pt") # (L, K, D)
    # L, K, D = codebook.shape
    # distance_map = get_distance_over_codebook(codebook)

    # for i in range(L):
    #     distance_df = pd.DataFrame(distance_map[i].numpy(), index=range(K), columns=range(K))
    #     distance_df.to_csv(f"codebook_distance_{i}.csv")

    for i in range(8):
        distance_matrix = pd.read_csv(f"distance/codebook_distance_{i}.csv")
        plot_distance_matrix(distance_matrix, title=f"Distance_{i}")
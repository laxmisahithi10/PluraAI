import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_image, split_lungs, generate_patches


class CALSLDataset(Dataset):
    def __init__(self, data, image_root):
        """
        data       : CSV file path OR pandas DataFrame
        image_root : root folder containing images
        """
        if isinstance(data, pd.DataFrame):
            self.df = data.reset_index(drop=True)
        else:
            self.df = pd.read_csv(data).reset_index(drop=True)

        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(10):  # try up to 10 times
            row = self.df.iloc[idx]

            image_name = row["image_path"]
            label = int(row["label"])

            image_path = os.path.join(self.image_root, image_name)

            try:
                img = preprocess_image(image_path)

                full_patches = generate_patches(img)
                left_img, right_img = split_lungs(img)
                left_patches = generate_patches(left_img)
                right_patches = generate_patches(right_img)

                return (
                    torch.tensor(full_patches, dtype=torch.float32),
                    torch.tensor(left_patches, dtype=torch.float32),
                    torch.tensor(right_patches, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long)
                )

            except Exception:
                # pick another random index if image is bad/missing
                idx = torch.randint(0, len(self.df), (1,)).item()

        # If all retries fail (very rare)
        raise RuntimeError("Too many corrupted/missing images encountered")

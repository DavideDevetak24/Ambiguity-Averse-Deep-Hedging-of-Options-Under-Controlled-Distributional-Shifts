"""
Script for dataset class with kmeans labels.
The function loader divides the dataset into batches and train, val and test
for training the neural network.
"""
# Using pytorch create dataloader for hedging dataset putting togeher paths and respective kmeans labels
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# We create a dataloader with the paths from data_s and the respective kmeans labels
class AADeepHedgingDataset(Dataset):
    def __init__(self, data_S, labels):
        super().__init__()
        self.paths = torch.tensor(
            data_S.values.T, dtype=torch.float32
        ).unsqueeze(-1) # shape (n_paths, n_steps, 1)

        self.labels = torch.tensor(labels, dtype=torch.long) # shape (n_paths,)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.paths[idx], self.labels[idx]


# Create loader function
def loader(dataset, batch_size=100, set_proportions = [0.6, 0.2, 0.2],
           shuffle=False, num_workers=0, pin_memory=None, persistent_workers=False):
    """
    The function loader divides the dataset into batches and train, val and test
    The input set_proportions is a list of three elements that sum to 1:
    they represent the proportions of train, val and test sets respectively.

    """

    assert sum(set_proportions) == 1.0, "set_proportions must sum to 1"
    assert len(set_proportions) == 3, "set_proportions must have three elements"

    n_total = len(dataset)
    n_train = int(set_proportions[0] * n_total)
    n_val = int(set_proportions[1] * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    persistent_workers = persistent_workers and num_workers > 0
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(train_set, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)
    test_loader = DataLoader(test_set, **loader_kwargs)

    return train_loader, val_loader, test_loader


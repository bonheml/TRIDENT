from torch.utils.data import Dataset


class WSIExplainerDataset(Dataset):
    """ Dataset from a WSI patcher to directly read tiles on a slide and return weights to compute loss for gradient rollout"""

    def __init__(self, patcher, transform, weights):
        self.patcher = patcher
        self.transform = transform
        self.weights = weights

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, index):
        tile, x, y = self.patcher[index]
        weight = self.weights[index]
        print(f"weights shape {weight.shape}")

        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y), weight
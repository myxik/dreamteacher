from torch.utils.data import Dataset


class HFDataset(Dataset):
    def __init__(self, dataset_obj, transforms):
        self.ds = dataset_obj
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.ds[idx]

        image = sample["image"]
        label = sample["label"]

        return self.transforms(image), label
    
    def __len__(self):
        return len(self.ds)

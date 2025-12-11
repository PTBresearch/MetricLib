from metric.data import Dataset
import pytest


def test_dataset_init():
    class TestDataset(Dataset):
        def __init__(self):
            pass

    dataset = TestDataset()
    with pytest.raises(NotImplementedError):
        dataset[0]


def test_dataset_get_metadata():
    class TestDataset(Dataset):
        def __init__(self):
            self.data = [
                (1, "A", {"meta1": "value1"}),
                (2, "B", {"meta1": "value2"}),
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    dataset = TestDataset()
    metadata = dataset.get_metadata()
    assert not metadata.empty
    assert list(metadata.columns) == ["meta1"]

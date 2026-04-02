from functools import lru_cache
import pandas as pd
from typing import Dict, Literal, TypeVar, Union, Tuple
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(TorchDataset[_T_co]):
    def __init__(
        self, name: str = None, metadata: pd.DataFrame = None, labels: pd.Series = None
    ):
        self.name = name
        self.metadata: pd.DataFrame = metadata
        self.labels: pd.Series = labels

    """
    Abstract base class for a custom PyTorch-compatible dataset.

    This class defines the basic interface for handling structured data
    along with associated metadata. Subclasses must implement `__getitem__`
    and `view_X`.
    """

    def __getitem__(self, index: int) -> Tuple[_T_co, Dict[str, Union[str, float]]]:
        """
        Retrieve a single data sample and its associated metadata.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple:
                A tuple containing:
                - `_T_co`: The data sample.
                - `Dict[str, Union[str, float]]`: Metadata dictionary, where
                  keys correspond to field names and values are metadata
                  attributes (strings or floats).
        """
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__")

    @lru_cache(maxsize=2, typed=False)
    def _get_label_and_metadata(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Internal helper method to collect labels and metadata for the dataset.

        This method iterates over all dataset entries and constructs a
        DataFrame containing labels and metadata. The result is cached
        for performance.

        Returns:
            tuple:
                A tuple containing:
                - `pd.Series`: Series of labels extracted from the dataset.
                - `pd.DataFrame`: DataFrame of metadata with one row per sample.
        """

        if (
            getattr(self, "metadata", None) is not None
            and getattr(self, "labels", None) is not None
        ):
            return self.labels, self.metadata

        data = pd.DataFrame(
            [
                dict({"_label": self[idx][1]}, **self[idx][2])
                for idx in tqdm(range(len(self)), "Loading dataset")
            ],
        )

        self.metadata = data.drop("_label", axis=1)
        self.labels = pd.Series(data["_label"])
        return pd.Series(data["_label"]), self.metadata

    def get_metadata(self) -> pd.DataFrame:
        """
        Retrieve the metadata for all samples in the dataset.

        Returns:
            pd.DataFrame: DataFrame containing metadata for all samples,
            with one row per sample and one column per metadata attribute.
        """
        return self._get_label_and_metadata()[1]

    def get_labels(self) -> pd.Series:
        """
        Retrieve the labels for all samples in the dataset.

        Returns:
            pd.Series: Series of labels corresponding to each sample.
        """
        return self._get_label_and_metadata()[0]

    def view_X(self, x: _T_co) -> bytes:
        """
        Convert a feature sample into a byte representation.

        This is typically used for visualization or serialization purposes.

        Args:
            x (_T_co): The data sample to convert.

        Returns:
            bytes: Byte representation of the feature sample.
        """
        raise NotImplementedError("Subclasses of Dataset should implement view_X")

    def to_pytorch(self) -> TorchDataset:
        """
        Convert this dataset into a standard PyTorch `Dataset` object.

        This is useful for direct compatibility with PyTorch `DataLoader`s.

        Returns:
            torch.utils.data.Dataset: A PyTorch-compatible dataset that yields
            `(data, metadata)` tuples.
        """

        class PyTorchDataset(TorchDataset):
            """Internal wrapper to make the custom dataset PyTorch-compatible."""

            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(
                self, index: int
            ) -> Tuple[_T_co, Dict[str, Union[str, float]]]:
                return self.dataset[index][0], self.dataset[index][1]

        return PyTorchDataset(self)

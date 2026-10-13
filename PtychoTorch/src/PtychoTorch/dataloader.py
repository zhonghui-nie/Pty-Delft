import random
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from itertools import compress

import torch
from torch.utils.data import BatchSampler

from PtychoTorch.propagators.base import BasePropagator
from PtychoTorch.typing import DiffPatternSet
from PtychoTorch.utils import dict_to_log_str

from .dataset import DataSet


class DataLoader(ABC):
    @property
    @abstractmethod
    def dataset(self) -> DataSet:
        raise NotImplementedError

    @abstractmethod
    def generator(
        self,
        propagator: BasePropagator,
        object_shape: torch.Size,
    ) -> Generator[list[DiffPatternSet]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class BasicDataLoader(DataLoader):
    def __init__(
        self,
        dataset: DataSet,
        randomize: bool = True,
        p: float = 1,
        barebeam_propability: float = 0.0,
    ):
        """
        Args:
            barebeam_propability: Propability with which barebeam is used instead of
                diffraction pattern.
        """
        self._sequence: Sequence[int] = []
        self.ds: DataSet = dataset

        self.randomize = randomize
        self.p = p
        self.barebeam_propability: float = barebeam_propability

        self.set_sequence_from_ds(randomize, p)

    @property
    def dataset(self) -> DataSet:
        return self.ds

    def __len__(self):
        return len(self.sequence)

    @property
    def sequence(self) -> list[int]:
        return list(self._sequence)

    def set_sequence(self, new_sequence: Sequence[int]):
        if max(new_sequence) >= len(self.ds.encoder):
            raise ValueError(
                "largest index of new_sequence must be smaller or equal than number of "
                "positions in BasicDataLoader.ds.encoder"
            )

        self._sequence = new_sequence

    def set_sequence_from_ds(self, randomize: bool = False, p: float = 1.0):
        """Set new random sequence.

        Args:
            randomize : Shuffle sequence order
            p : propability to draw item.

        """
        new_seq = list(range(len(self.ds)))
        if randomize:
            random.shuffle(new_seq)

        N = len(new_seq)
        mask: list[int]
        mask = (torch.linspace(1 / N, 1, N) <= p).int().tolist()  # type: ignore
        random.shuffle(mask)

        self.set_sequence(list(compress(new_seq, mask)))

    def generator(
        self,
        propagator: BasePropagator,
        object_shape: torch.Size | None = None,
    ) -> Generator[list[DiffPatternSet]]:
        propagator.check_compatibility_with_dataset(self.ds)

        # reshuffle sequence
        if self.randomize:
            self.set_sequence_from_ds(self.randomize, self.p)

        if max(self.sequence) >= len(self.ds):
            raise ValueError(
                "BasicDataLoader.ds must have at least as many diffraction patterns as "
                "the max index of self.sequence"
            )

        barebeam = self.ds.get_barebeam()
        if barebeam is None and self.barebeam_propability > 0:
            raise Exception(
                f"No barebeam found, but {self.barebeam_propability = } != 0"
            )

        for shuffled_index in self.sequence:
            if random.random() < self.barebeam_propability:
                if barebeam is None:
                    raise Exception("No barebeam found.")
                yield [barebeam]
                continue

            diffraction_pattern_set: DiffPatternSet = self.ds[shuffled_index]
            if (scan_pos_SI := diffraction_pattern_set.scan_position_SI) is None:
                raise ValueError(
                    f"scan_position_SI is None. Is self.ds[{shuffled_index}] barebeam?"
                )
            scan_position_px = self.ds.encoder.get_position_px(
                scan_position_SI=scan_pos_SI,
                prop=propagator,
                object_shape=object_shape,
            )
            diffraction_pattern_set.scan_position_px = scan_position_px
            yield [diffraction_pattern_set]

    def __str__(self) -> str:
        title = f"DataLoader: {self.__class__.__name__}"

        key_dict = {
            "N_scanningpositions": f"{len(self)}",
            "randomize": self.randomize,
            "p": self.p,
        }
        ret_str = dict_to_log_str(title, key_dict)

        ds_str = "\n\t".join(["", *str(self.ds).split("\n")])

        return ret_str + ds_str


class BatchDataLoader(DataLoader):
    def __init__(
        self,
        ds: DataSet,
        batch_size: int,
        drop_last: bool = False,
        barebeam_propability: float = 0,
    ) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.barebeam_propability = barebeam_propability

    def _get_batch_sampler(self) -> Generator[list[DiffPatternSet], None, None]:
        return BatchSampler(self.ds, self.batch_size, self.drop_last)  # pyright: ignore

    def __len__(self) -> int:
        bs = self._get_batch_sampler()
        return sum(1 for _ in bs)

        # return len(bs) would also work, but typing wouldn't allow it.

    @property
    def dataset(self):
        return self.ds

    def generator(
        self, propagator: BasePropagator, object_shape: torch.Size
    ) -> Generator[list[DiffPatternSet], None, None]:
        bs = self._get_batch_sampler()
        barebeam = self.ds.get_barebeam()

        if barebeam is None and self.barebeam_propability > 0:
            raise Exception(
                f"No barebeam found, but {self.barebeam_propability = } != 0"
            )

        for diff_pattern_set_batch in bs:
            for idx, diff_pattern in enumerate(diff_pattern_set_batch):
                if diff_pattern.scan_position_SI is None:
                    raise Exception("scan_position_SI is None")

                if random.random() < self.barebeam_propability:
                    if barebeam is None:
                        raise Exception("No barebeam found")
                    diff_pattern_set_batch[idx] = barebeam

                scan_position_px = self.ds.encoder.get_position_px(
                    scan_position_SI=diff_pattern.scan_position_SI,
                    prop=propagator,
                    object_shape=object_shape,
                )
                diff_pattern.scan_position_px = scan_position_px

            yield diff_pattern_set_batch

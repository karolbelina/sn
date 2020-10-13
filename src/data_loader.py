from collections.abc import Iterator

import numpy as np
import random


class DataLoader:
    def __init__(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
        batch_size: int = None,
        random: bool = True,
    ) -> None:
        self.data = data
        self.batch_size = batch_size if batch_size is not None else len(data)
        self.random = random

    def get_batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.random:
            random.shuffle(self.data)

        for i in range(0, len(self.data), self.batch_size):
            data = self.data[i: i + self.batch_size]
            x, y_hat = zip(*data)
            x = np.vstack(x)
            y_hat = np.vstack(y_hat)
            yield x, y_hat

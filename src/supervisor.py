from data_loader import DataLoader
from model import Model
from trainer import Trainer
from typing import Optional

import numpy as np
import pickle as pkl


class Supervisor:
    def __init__(
        self,
        model: Model,
        trainer: Trainer,
        max_epochs: Optional[int] = None,
        epsilon: float = 0
    ) -> None:
        self._max_epochs = max_epochs if max_epochs is not None else np.inf
        self._epsilon = epsilon

        self._error = None
        self._epoch_number = 0
        
        self._model = model
        self._trainer = trainer

    def run(self) -> int:
        try:
            while self._epoch_number < self._max_epochs:
                self._epoch_number += 1

                self._error = self._trainer.fit(self._model)

                if self._error <= self._epsilon:
                    print("Achieved the derired error threshold")
                    self.pause()
                    break

            return self._epoch_number
        except KeyboardInterrupt:
           self.pause()
    
    def pause(self) -> None:
        while True:
            try:
                decision = input("\nDo you want to save the state? (y/N): ")
                if decision.strip() == 'y':
                    path = input("Please enter the file path: ")
                    with open(path, 'wb') as file:
                        pkl.dump(self, file)
                        print(f"Saved the state in {path}")
                    return
                elif decision.strip() == 'N':
                    return
            except KeyboardInterrupt:
                return
           
    @staticmethod
    def resume(filename: str):
        with open(filename, 'rb') as file:
            return pkl.load(file)

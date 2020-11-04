from copy import deepcopy
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
        epsilon: float = 0,
        early_stopping_threshold: float = np.inf
    ) -> None:
        self._max_epochs = max_epochs if max_epochs is not None else np.inf
        self._epsilon = epsilon
        self._early_stopping_threshold = early_stopping_threshold

        self._best_model = deepcopy(model)
        self._best_error = np.inf

        self._error = np.inf
        self._epoch_number = 0
        
        self._model = model
        self._trainer = trainer

    def __call__(self, silent: bool = False) -> int:
        def callback(epoch_report: object) -> None:
            if not silent:
                loss = epoch_report['val_loss']
                accuracy = epoch_report['val_accuracy']
                print(f"loss = {loss:.2f}, acc = {accuracy:.2%}")

        try:
            while self._epoch_number < self._max_epochs:
                self._error = self._trainer.fit(self._model, callback=callback)

                if self._error <= self._epsilon:
                    if not silent:
                        print("Reached the derired error threshold")
                    break
                
                if self._error < self._best_error:
                    self._best_error = self._error
                    self._best_model = deepcopy(self._model)

                error_delta = self._error - self._best_error
                if error_delta >= self._early_stopping_threshold:
                    self._model = self._best_error
                    if not silent:
                        print(f"Reached the early stopping threshold ({error_delta:.2%})")
                        print("Backing up...")
                    break

                self._epoch_number += 1
                if self._epoch_number >= self._max_epochs:
                    if not silent:
                        print("Reached the maximum number of epochs")
                    break

            if not silent:
                self.pause()

            return self._epoch_number
        except KeyboardInterrupt:
            if not silent:
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

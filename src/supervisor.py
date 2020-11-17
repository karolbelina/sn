from copy import copy, deepcopy
from data_loader import DataLoader
from enum import Enum
from model import Model
from trainer import Trainer
from typing import Optional

import numpy as np
import pickle as pkl


class StopReason(Enum):
    THRESHOLD_REACHED = 1
    MAX_EPOCHS = 2
    EARLY_STOPPING = 3


class ResearchSupervisor:
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

        self._best_error = np.inf
        self._best_report = None

        self._model = model
        self._error = np.inf

        self._epoch_accuracy_threshold = np.inf
        self._epoch_number = 0
        
        self._trainer = trainer

    def __call__(self) -> tuple[int, StopReason]:
        while True:
            self._epoch_number += 1

            self._trainer.fit(self._model)
            self._error = self._trainer.validate(self._model)

            print(f"epoch {self._epoch_number}, accuracy = {1 - self._error:.2%}")

            if self._error <= self._epsilon and np.isinf(self._epoch_accuracy_threshold):
                self._epoch_accuracy_threshold = self._epoch_number
            
            if self._error < self._best_error:
                self._best_error = self._error

            error_delta = self._error - self._best_error
            if error_delta >= self._early_stopping_threshold:
                self._error = self._best_error
                return self._epoch_accuracy_threshold, 1 - self._best_error

            if self._epoch_number >= self._max_epochs:
                self._error = self._best_error
                return self._epoch_accuracy_threshold, 1 - self._best_error


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
        self._best_report = None

        self._model = model
        self._error = np.inf
        self._report = None

        self._epoch_number = 0
        
        self._trainer = trainer

    def __call__(self, silent: bool = False) -> tuple[int, StopReason]:
        try:
            while True:
                self._epoch_number += 1

                self._trainer.fit(self._model)
                self._error, self._report = self._trainer.validate(self._model)

                if not silent:
                    loss = self._report['val_loss']
                    accuracy = self._report['val_accuracy']
                    print(f"loss = {loss:.2f}, acc = {accuracy:.2%}")

                if self._error <= self._epsilon:
                    if not silent:
                        print("Reached the derired error threshold")
                        self.pause()
                    return self._epoch_number, self._report, StopReason.THRESHOLD_REACHED
                
                if self._error < self._best_error:
                    self._best_error = self._error
                    self._best_model = deepcopy(self._model)
                    self._best_report = self._report

                error_delta = self._error - self._best_error
                if error_delta >= self._early_stopping_threshold:
                    self._model = self._best_error
                    self._report = self._best_report
                    if not silent:
                        print(f"Reached the early stopping threshold ({error_delta:.2%})")
                        print("Backing up...")
                    return self._epoch_number, self._report, StopReason.EARLY_STOPPING

                if self._epoch_number >= self._max_epochs:
                    if not silent:
                        print("Reached the maximum number of epochs")
                    return self._epoch_number, self._report, StopReason.MAX_EPOCHS
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

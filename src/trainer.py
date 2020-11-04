from abc import ABC, abstractmethod
from data_loader import DataLoader
from model import Model
from typing import Optional


class Trainer(ABC):
    @abstractmethod
    def fit(self, model: Model, callback: callable) -> float:
        pass

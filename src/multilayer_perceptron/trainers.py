from .module import MultilayerPerceptron
from abc import abstractmethod
from data_loader import DataLoader
from model import Model
from trainer import Trainer
from typing import Optional

import numpy as np


class AbstractTrainer(Trainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ) -> None:
        self._train_dataloader = train_dataloader
        if val_dataloader is None:
            self._val_dataloader = train_dataloader
        else:
            self._val_dataloader = val_dataloader

    @abstractmethod
    def fit(self, model: MultilayerPerceptron) -> None:
        pass
    
    def validate(self, model: Model) -> tuple[float, object]:
        x, y = self._val_dataloader.get_data()
        y_hat = model(x)

        result_classes = y_hat.argmax(axis=1)
        label_classes = y.argmax(axis=1)

        Nc = (result_classes == label_classes).sum()
        Nt = len(result_classes)
        
        error = (Nt - Nc) / Nt

        return error


class SGDTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._α = learning_rate

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        α = self._α

        for x, y in self._train_dataloader.get_batches():
            dC_dθ = model.generate_dC_dθ(x, y)

            θ = θ - α * dC_dθ(θ)
        
        model.θ = θ


class SGDMomentumTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        momentum_factor: float = 0.8,
        learning_rate: float = 1e-2
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._γ = momentum_factor
        self._α = learning_rate

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        α = self._α
        γ = self._γ
        v = np.zeros_like(θ)

        for x, y in self._train_dataloader.get_batches():
            dC_dθ = model.generate_dC_dθ(x, y)

            v = γ * v + α * dC_dθ(θ)
            θ = θ - v
        
        model.θ = θ


class NesterovMomentumTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        momentum_factor: float = 0.8,
        learning_rate: float = 1e-2
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._γ = momentum_factor
        self._α = learning_rate

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        α = self._α
        γ = self._γ
        v = np.zeros_like(θ)

        for x, y in self._train_dataloader.get_batches():
            dC_dθ = model.generate_dC_dθ(x, y)

            v = γ * v + α * dC_dθ(θ - γ * v)
            θ = θ - v
        
        model.θ = θ

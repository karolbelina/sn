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
    
    @staticmethod
    def loss(output: np.ndarray, label: np.ndarray) -> np.ndarray:
        stable_output = np.clip(output, 1e-12, None)
        cross_e = (-np.log(stable_output) * label).sum(axis=1)

        return cross_e.mean()
    
    def validate(self, model: Model) -> tuple[float, object]:
        x, y = self._val_dataloader.get_data()
        y_hat = model(x)

        loss = AbstractTrainer.loss(y_hat, y)

        result_classes = y_hat.argmax(axis=1)
        label_classes = y.argmax(axis=1)

        Nc = (result_classes == label_classes).sum()
        Nt = len(result_classes)
        
        error = (Nt - Nc) / Nt

        return error, loss


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
            θ = θ - α * model.dC_dθ(x, y)(θ)
        
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
            v = γ * v + α * model.dC_dθ(x, y)(θ)
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
            v = γ * v + α * model.dC_dθ(x, y)(θ - γ * v)
            θ = θ - v
        
        model.θ = θ


class AdagradTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-2
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._α = learning_rate

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        α = self._α
        ε = 1e-8
        g = np.zeros_like(θ)

        for x, y in self._train_dataloader.get_batches():
            dC_dθ_θ = model.dC_dθ(x, y)(θ)

            g = g + np.square(dC_dθ_θ)

            θ = θ - α / np.sqrt(ε + g) * dC_dθ_θ
        
        model.θ = θ


class AdadeltaTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-2,
        decay: float = 0.99
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._γ = decay

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        γ = self._γ
        ε = 1e-8
        g = np.zeros_like(θ)
        s = np.zeros_like(θ)

        for x, y in self._train_dataloader.get_batches():
            dC_dθ_θ = model.dC_dθ(x, y)(θ)

            g = (1 - γ) * np.square(dC_dθ_θ) + γ * g
            Δθ = -np.sqrt(s + ε) / np.sqrt(g + ε) * dC_dθ_θ
            s = (1 - γ) * np.square(Δθ) + γ * s
            θ = θ + Δθ
            
        model.θ = θ


class AdamTrainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.999
    ) -> None:
        super().__init__(train_dataloader, val_dataloader)
        self._α = learning_rate
        self._β1 = beta_1
        self._β2 = beta_2

    def fit(self, model: MultilayerPerceptron) -> None:
        θ = model.θ
        α = self._α
        β1 = self._β1
        β2 = self._β2
        ε = 1e-8
        t = 0
        m = np.zeros_like(θ)
        v = np.zeros_like(θ)

        for x, y in self._train_dataloader.get_batches():
            t = t + 1

            g = model.dC_dθ(x, y)(θ)

            m = β1 * m + (1 - β1) * g
            v = β2 * v + (1 - β2) * np.square(g)

            m_hat = m / (1 - β1 ** t)
            v_hat = v / (1 - β2 ** t)

            θ = θ - α / (np.sqrt(v_hat) + ε) * m_hat
            
        model.θ = θ

from .module import MultilayerPerceptron
from data_loader import DataLoader
from layers import Layer
from model import Model
from trainer import Trainer as AbstractTrainer
from typing import Optional

import numpy as np


class Trainer(AbstractTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        self._learning_rate = learning_rate
        self._train_dataloader = train_dataloader
        if val_dataloader is None:
            self._val_dataloader = train_dataloader
        else:
            self._val_dataloader = val_dataloader
    
    @staticmethod
    def _cost_fn(output: np.ndarray, label: np.ndarray) -> np.ndarray:
        stable_output = np.clip(output, 1e-12, None)
        cross_e = (-np.log(stable_output) * label).sum(axis=1)

        return cross_e.mean()
    
    @staticmethod
    def _cost_fn_backward(output: np.ndarray, label: np.ndarray) -> np.ndarray:
        stable_output = np.clip(output, 1e-12, None)

        return stable_output - label

    def fit(self, model: MultilayerPerceptron) -> None:
        for x, y in self._train_dataloader.get_batches():
            y_hat = model(x)
            da = self._cost_fn_backward(y_hat, y)

            for layer in reversed(model._layers):
                dw, db, da = layer.backpropagate(da)
                self._update_layer_weights(layer, dw, db)
        
        val_error = self._validate(model, next(self._val_dataloader.get_batches()))

        return val_error
    
    def _validate(self, model: Model, val_batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y_hat = val_batch
        y = model(x)
        error = self._cost_fn(y, y_hat)

        result_classes = y_hat.argmax(axis=1)
        label_classes = y.argmax(axis=1)
        acc = (result_classes == label_classes).mean()

        print(f"loss = {error}, acc = {acc * 100}%")

        return error
    
    def _update_layer_weights(self, layer: Layer, dw: np.ndarray, db: np.ndarray):
        layer.update_weights(self._learning_rate * dw)
        layer.update_biases(self._learning_rate * db)

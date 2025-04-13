from dataclasses import dataclass

@dataclass
class Config:
    learning_rate: float = 0.01
    momentum: float = 0.0
    epochs: int = 1000
    batch_size: int = 64
    gmin: float = 0.001
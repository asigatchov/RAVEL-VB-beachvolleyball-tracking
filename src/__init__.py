"""PyTorch inference implementation for the TAPe volleyball detector."""

from .model import TAPeVB2Config, TAPeVB2Model, validate_checkpoint_architecture

TAPeVBConfig = TAPeVB2Config
TAPeVBModel = TAPeVB2Model

__all__ = [
    "TAPeVB2Config",
    "TAPeVB2Model",
    "TAPeVBConfig",
    "TAPeVBModel",
    "validate_checkpoint_architecture",
]

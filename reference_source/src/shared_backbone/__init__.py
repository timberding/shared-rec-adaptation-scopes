"""Shared-backbone model surfaces for DeltaGate resubmission experiments."""

from .model import (
    DomainItemHead,
    SharedBackboneDeltaGateAdapter,
    SharedGRUEncoder,
    SharedSASRecEncoder,
    freeze_module,
    trainable_parameter_report,
)

__all__ = [
    "DomainItemHead",
    "SharedBackboneDeltaGateAdapter",
    "SharedGRUEncoder",
    "SharedSASRecEncoder",
    "freeze_module",
    "trainable_parameter_report",
]

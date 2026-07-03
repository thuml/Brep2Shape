"""Public Brep2Shape model API."""

from .config import Brep2ShapeConfig
from .dual_classification import DualClassification
from .dual_encoder import DualCurveEncoder, DualGraphEncoder, DualSurfaceEncoder
from .dual_segmentation import DualSegmentation
from .encoders import PredictionHead
from .pretraining import UVPointPrediction


__all__ = [
    "Brep2ShapeConfig",
    "DualClassification",
    "DualCurveEncoder",
    "DualGraphEncoder",
    "DualSegmentation",
    "DualSurfaceEncoder",
    "PredictionHead",
    "UVPointPrediction",
]

"""
engines/ml_engine.py — compatibility shim.

The prediction pipeline lives in app/engines/ml/ now, split by stage:
targets decides what is worth predicting, features builds the matrix,
training fits and scores, importance explains which columns mattered,
whatif answers a single hypothetical, insights writes it up, and runner
sequences all of it. It had grown to 1,022 lines covering every one of
those, with the section banners inside already marking the seams.

This keeps `from app.engines.ml_engine import run_ml_pipeline` working.
What it forwards is what something actually imports — checked against
the codebase. New code should import from app.engines.ml.
"""
from app.engines.ml.results import (                            # noqa: F401
    FeatureImportance, MLReport, ModelResult,
)
from app.engines.ml.targets import (                            # noqa: F401
    detect_task, suggest_targets,
)
from app.engines.ml.features import (                           # noqa: F401
    MAX_ONEHOT_LEVELS, prepare_features,
)
from app.engines.ml.training import train_models                # noqa: F401
from app.engines.ml.importance import get_feature_importance    # noqa: F401
from app.engines.ml.whatif import predict_what_if               # noqa: F401
from app.engines.ml.runner import run_ml_pipeline               # noqa: F401

__all__ = ["run_ml_pipeline", "predict_what_if", "MLReport"]

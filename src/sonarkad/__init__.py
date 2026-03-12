"""SonarKAD: physics-aligned Kolmogorov–Arnold Network utilities.

The codebase implements an interpretable additive decomposition aligned with the
(passive) sonar equation in the log-intensity domain.

- ``SonarKAD``: additive-only model (interaction_rank=0)
- ``SonarKAD``: same backbone with a low-rank nonseparable interaction enabled via
  ``interaction_rank>0`` (alias for clarity)
"""

from .models import SonarKAD, SonarKADConfig, SonarKAD, SmallMLP  # noqa: F401

# Convenience exports for deployment
from .deploy import load_sonarkad_model_bundle, predict_from_bundle, predict_rl  # noqa: F401

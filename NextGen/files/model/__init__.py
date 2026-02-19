# files/model/__init__.py
from .world_model import WorldModel
from .self_model import SelfModel

# TODO: Import these modules when they are implemented
# from .agency_model import AgencyModel
# from .meta_cognitive import MetaCognitiveModel
# from .consciousness import ConsciousnessIntegrator
# from .behavior import BehaviorGenerator
# from .self_aware_ai import SelfAwareAI

__all__ = [
    'WorldModel', 'SelfModel',  # 'AgencyModel',
    # 'MetaCognitiveModel', 'ConsciousnessIntegrator',
    # 'BehaviorGenerator', 'SelfAwareAI'
]

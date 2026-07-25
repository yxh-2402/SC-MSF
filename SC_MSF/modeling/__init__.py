__all__ = ['make_model']

from SC_MSF.modeling.PIE import PIE
from SC_MSF.modeling.JAAD import JAAD

_MODELS_ = {
    'JAAD': JAAD,
    'PIE': PIE
}


def make_model(cfg):
    try:
        model = _MODELS_[cfg.METHOD]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model method {cfg.METHOD!r}; choose one of {sorted(_MODELS_)}"
        ) from exc
    return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)

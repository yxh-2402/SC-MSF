from SC_MSF.engine.trainer import do_train
from SC_MSF.engine.trainer import do_val
from SC_MSF.engine.trainer import inference

ENGINE_ZOO = {
                'JAAD': (do_train, do_val, inference),
                'PIE': (do_train, do_val, inference),
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]

"""Numerical constants of the reported RC-F and joint alignment objectives."""
PACIFICA = (2.297, 4.049, 1.127)
LOCAL_WEIGHT, CLEAN_WEIGHT, STATE_WEIGHT, TERMINAL_WEIGHT = .1, .5, .25, 1.
METRIC_SCALES = (1., .1, .02)
NATIVE_SCALES = (33.37, 21., 1.765)
CLEAN_STEP_WEIGHTS = (.1, .2, .3, .4)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def lora_configuration():
    return dict(language_layers=36, layers=list(range(28,36)), targets=["q_proj","v_proj"],
                rank=16, alpha=32, dropout=0., initialization="A-kaiming-B-zero")

def make_late_dit_config():
    return dict(self_attention_layers=[12,14], ffn_layers=[12,13,14,15], adaln_layers=[12,13,14,15])

"""disco_torch: PyTorch port of DeepMind's Disco103 meta-learned RL update rule."""

from disco_torch.types import UpdateRuleInputs, ValueOuts, EmaState, MetaNetInputOption, TransformConfig
from disco_torch.update_rule import DiscoUpdateRule
from disco_torch.load_weights import load_disco103_weights, download_disco103_weights
from disco_torch.adapter import DiscoAdapter, extract_credit, credit_from_lm_rollout
from disco_torch.grpo import grpo_advantages, modulate_with_credit

__all__ = [
    "DiscoUpdateRule",
    "UpdateRuleInputs",
    "ValueOuts",
    "EmaState",
    "MetaNetInputOption",
    "TransformConfig",
    "load_disco103_weights",
    "download_disco103_weights",
    "DiscoAdapter",
    "extract_credit",
    "credit_from_lm_rollout",
    "grpo_advantages",
    "modulate_with_credit",
]

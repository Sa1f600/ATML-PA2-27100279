from __future__ import annotations


def locate_transformer_backbone(value_model):
    """Return the transformer inside a sequence-classification value model.

    PEFT wraps the complete sequence classifier, not just its transformer.
    Calling that wrapper would run the scalar ``score``/``classifier`` head as
    part of the forward pass.  PPO needs the token-level hidden states instead,
    so unwrap a PEFT model to its classifier and then select that classifier's
    declared transformer backbone.

    This helper intentionally uses the public ``get_base_model`` and
    ``base_model_prefix`` interfaces rather than depending on PEFT's private
    wrapper nesting.
    """
    get_base_model = getattr(value_model, "get_base_model", None)
    classifier_model = (
        get_base_model() if callable(get_base_model) else value_model
    )
    backbone_name = getattr(classifier_model, "base_model_prefix", None)
    backbone = (
        getattr(classifier_model, backbone_name, None)
        if isinstance(backbone_name, str) and backbone_name
        else None
    )
    if backbone is None or backbone is classifier_model:
        raise RuntimeError(
            "Could not locate the transformer backbone inside the value model"
        )
    return backbone

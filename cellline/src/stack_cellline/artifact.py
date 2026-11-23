"""Save and reload a fine-tuned STACK: adapters on top of the released checkpoint.

The file holds the same three entries as the shared adapter of ttt/scripts/norman_ttt.py:
  adapters  every LoRA (A, B) pair, in model.modules() order (stack_ttt.lora.adapter_state)
  cls       the co-trained unmasking classifier f_CLS (a full state dict)
  settings  plain values only: the format tag, the adapted layer names, rank, alpha, and
            how the adapters were trained

The base weights are not in the file. Reloading needs the same base checkpoint, and
load_finetuned injects the adapters into exactly the layers named in the file.
Everything is read with torch.load(weights_only=True).
"""

import torch
from stack_ttt.lora import adapter_state, inject_lora, load_adapter_state
from stack_ttt.model import load_stack

FORMAT = "stack-cellline-lora/1"


def save_adapter(path, model, settings: dict) -> None:
    """Write the model's adapters and f_CLS, with `settings`, which must name the adapted
    layers ("targets"), "rank" and "alpha"."""
    for key in ("targets", "rank", "alpha"):
        if key not in settings:
            raise ValueError(f"settings need {key!r} to reload the adapters")
    torch.save(
        {
            "adapters": [(a.cpu(), b.cpu()) for a, b in adapter_state(model)],
            "cls": {k: v.detach().cpu().clone() for k, v in model.cls.state_dict().items()},
            "settings": {"format": FORMAT, **settings},
        },
        path,
    )


def read_settings(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["settings"]


def load_finetuned(checkpoint, adapter=None, device="cpu"):
    """STACK for inference: zero-shot when `adapter` is None, else fine-tuned.

    Zero-shot is the released checkpoint unchanged, with its own f_CLS. Do not get it by
    wrapping a fine-tuned model in stack_ttt.lora.adapters_disabled: that turns the
    adapters off but keeps the fine-tuned f_CLS.

    Both come back in eval mode with every parameter frozen, so the two arms run in the
    same numerical state. On CPU, whether parameters require gradients changes STACK's
    embeddings by about 1e-6 even under torch.no_grad, which energy distance can amplify
    to about 1e-4 relative.
    """
    model, _ = load_stack(checkpoint, device=device)
    if adapter is not None:
        saved = torch.load(adapter, map_location=device, weights_only=True)
        settings = saved["settings"]
        if settings.get("format") != FORMAT:
            raise ValueError(f"{adapter} is not a {FORMAT} file: format {settings.get('format')!r}")
        inject_lora(model, settings["targets"], rank=settings["rank"], alpha=settings["alpha"])
        load_adapter_state(model, saved["adapters"])
        model.cls.load_state_dict(saved["cls"])
    return model.requires_grad_(False).eval()

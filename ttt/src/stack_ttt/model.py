"""Load a STACK fine-tuning checkpoint without running pickled code."""

import torch
from stack.model_finetune import ICL_FinetunedModel


def load_stack(checkpoint, device="cpu"):
    """The ICL_FinetunedModel in eval mode, and the checkpoint's hyper-parameters.

    torch.load runs with weights_only=True, which rebuilds only tensors and plain
    containers (arc-stack's own loader unpickles whatever the file holds), and with
    mmap=True, so tensors the model does not use, such as optimizer state, stay on disk.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    hparams = dict(ckpt["hyper_parameters"])
    model = ICL_FinetunedModel(**hparams["model_config"])
    model.load_state_dict({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), hparams

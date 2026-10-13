import torch


def optimizer_to_str(optim: torch.optim.Optimizer) -> str:
    """Summarize Optimizer to yaml-like string.

    Args:
        optim: Optimizer

    Returns:
        str: Description of optim

    """
    sep = "\t"

    name = (
        "Optimizer: " + optim.__class__.__module__ + "." + optim.__class__.__qualname__
    )
    ret = name

    param_groups = optim.state_dict().get("param_groups", [])
    for i, pg in enumerate(param_groups):
        ret += f"\n{sep}param_group_{i}:"

        for key, val in pg.items():
            ret += f"\n{sep}{sep}{key}={val}"

    return ret

from . import repr_tools

__all__ = ["repr_tools", "ROMEHyperParams", "apply_rome_to_model", "execute_rome"]


def __getattr__(name):
    if name == "ROMEHyperParams":
        from .hparams import ROMEHyperParams

        return ROMEHyperParams
    if name in {"apply_rome_to_model", "execute_rome"}:
        from .editor import apply_rome_to_model, execute_rome

        return {
            "apply_rome_to_model": apply_rome_to_model,
            "execute_rome": execute_rome,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

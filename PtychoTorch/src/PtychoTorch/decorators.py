from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

if TYPE_CHECKING:
    from .engine import AutoGradPIE
else:
    AutoGradPIE = Any


P = ParamSpec("P")
R = TypeVar("R")


def is_reconstruct[**P, R](
    func: Callable[Concatenate[AutoGradPIE, P], R | None],
) -> Callable[Concatenate[AutoGradPIE, P], R | None]:
    @wraps(func)
    def inner(
        self: AutoGradPIE,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | None:
        self.is_reconstructing = True

        epochs_before = self.epoch_counter
        interrupted: bool = True

        ret: R | None = None
        try:
            ret = func(self, *args, **kwargs)
            interrupted = False
        except KeyboardInterrupt:
            pass
        finally:
            delta_epochs = self.epoch_counter - epochs_before
            self.is_reconstructing = False

            if interrupted:
                self.log_ln(f"Reconstruction interrupted after +{delta_epochs} epochs.")
            else:
                self.log_ln(f"Reconstruction completed after +{delta_epochs} epochs.")

            self.log_ln(f"Total epochs: {self.epoch_counter}")
            self.log_sep()

        return ret

    return inner

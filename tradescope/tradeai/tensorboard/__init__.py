# ensure users can still use a non-torch tradeai version
try:
    from tradescope.tradeai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from tradescope.tradeai.tensorboard.base_tensorboard import (BaseTensorBoardCallback,
                                                                 BaseTensorboardLogger)
    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = (
    "TBLogger",
    "TBCallback"
)

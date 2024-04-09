# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for tradeai
"""
import logging
from pathlib import Path

from tradescope.constants import USERPATH_TRADEAIMODELS, Config
from tradescope.exceptions import OperationalException
from tradescope.resolvers import IResolver
from tradescope.tradeai.tradeai_interface import ITradeaiModel


logger = logging.getLogger(__name__)


class TradeaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = ITradeaiModel
    object_type_str = "TradeaiModel"
    user_subdir = USERPATH_TRADEAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("tradeai/prediction_models").resolve()
    )
    extra_path = "tradeaimodel_path"

    @staticmethod
    def load_tradeaimodel(config: Config) -> ITradeaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        tradeaimodel_name = config.get("tradeaimodel")
        if not tradeaimodel_name:
            raise OperationalException(
                "No tradeaimodel set. Please use `--tradeaimodel` to "
                "specify the TradeaiModel class to use.\n"
            )
        if tradeaimodel_name in disallowed_models:
            raise OperationalException(
                f"{tradeaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        tradeaimodel = TradeaiModelResolver.load_object(
            tradeaimodel_name,
            config,
            kwargs={"config": config},
        )

        return tradeaimodel

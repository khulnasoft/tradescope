""" Bitmart exchange subclass """
import logging
from typing import Dict

from tradescope.exchange import Exchange


logger = logging.getLogger(__name__)


class Bitmart(Exchange):
    """
    Bitmart exchange class. Contains adjustments needed for Tradescope to work
    with this exchange.
    """

    _ts_has: Dict = {
        "stoploss_on_exchange": False,  # Bitmart API does not support stoploss orders
        "ohlcv_candle_limit": 200,
    }

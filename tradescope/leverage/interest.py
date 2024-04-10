from math import ceil

from tradescope.exceptions import OperationalException
from tradescope.util import TsPrecise


one = TsPrecise(1.0)
four = TsPrecise(4.0)
twenty_four = TsPrecise(24.0)


def interest(
    exchange_name: str,
    borrowed: TsPrecise,
    rate: TsPrecise,
    hours: TsPrecise
) -> TsPrecise:
    """
    Equation to calculate interest on margin trades

    :param exchange_name: The exchanged being trading on
    :param borrowed: The amount of currency being borrowed
    :param rate: The rate of interest (i.e daily interest rate)
    :param hours: The time in hours that the currency has been borrowed for

    Raises:
        OperationalException: Raised if tradescope does
        not support margin trading for this exchange

    Returns: The amount of interest owed (currency matches borrowed)
    """
    exchange_name = exchange_name.lower()
    if exchange_name == "binance":
        return borrowed * rate * TsPrecise(ceil(hours)) / twenty_four
    elif exchange_name == "kraken":
        # Rounded based on https://kraken-fees-calculator.github.io/
        return borrowed * rate * (one + TsPrecise(ceil(hours / four)))
    else:
        raise OperationalException(f"Leverage not available on {exchange_name} with tradescope")

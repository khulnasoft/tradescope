from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from tests.conftest import (CURRENT_TEST_STRATEGY, get_args, get_patched_exchange, log_has_re,
                            patch_exchange, patched_configuration_load_config_file)
from tests.tradeai.conftest import get_patched_tradeai_strategy
from tradescope.commands.optimize_commands import setup_optimize_configuration
from tradescope.configuration.timerange import TimeRange
from tradescope.data import history
from tradescope.data.dataprovider import DataProvider
from tradescope.enums import RunMode
from tradescope.enums.candletype import CandleType
from tradescope.exceptions import OperationalException
from tradescope.optimize.backtesting import Backtesting
from tradescope.tradeai.data_kitchen import TradeaiDataKitchen


def test_tradeai_backtest_start_backtest_list(tradeai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('tradescope.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('tradescope.optimize.backtesting.history.load_data')
    mocker.patch('tradescope.optimize.backtesting.history.get_timerange', return_value=(now, now))

    patched_configuration_load_config_file(mocker, tradeai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--strategy-list', CURRENT_TEST_STRATEGY
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re('Using --strategy-list with TradeAI REQUIRES all strategies to have identical',
                      caplog)
    Backtesting.cleanup()


@pytest.mark.parametrize(
    "timeframe, expected_startup_candle_count",
    [
        ("5m", 876),
        ("15m", 492),
        ("1d", 302),
    ],
)
def test_tradeai_backtest_load_data(tradeai_conf, mocker, caplog,
                                   timeframe, expected_startup_candle_count):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('tradescope.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('tradescope.optimize.backtesting.history.load_data')
    mocker.patch('tradescope.optimize.backtesting.history.get_timerange', return_value=(now, now))
    tradeai_conf['timeframe'] = timeframe
    tradeai_conf.get('tradeai', {}).get('feature_parameters', {}).update({'include_timeframes': []})
    backtesting = Backtesting(deepcopy(tradeai_conf))
    backtesting.load_bt_data()

    assert log_has_re(f'Increasing startup_candle_count for tradeai on {timeframe} '
                      f'to {expected_startup_candle_count}', caplog)
    assert history.load_data.call_args[1]['startup_candles'] == expected_startup_candle_count

    Backtesting.cleanup()


def test_tradeai_backtest_live_models_model_not_found(tradeai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('tradescope.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('tradescope.optimize.backtesting.history.load_data')
    mocker.patch('tradescope.optimize.backtesting.history.get_timerange', return_value=(now, now))
    tradeai_conf["timerange"] = ""
    tradeai_conf.get("tradeai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, tradeai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--tradeai-backtest-live-models'
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(OperationalException,
                       match=r".* Historic predictions data is required to run backtest .*"):
        Backtesting(bt_config)

    Backtesting.cleanup()


def test_tradeai_backtest_consistent_timerange(mocker, tradeai_conf):
    tradeai_conf['runmode'] = 'backtest'
    mocker.patch('tradescope.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['XRP/USDT:USDT']))

    gbs = mocker.patch('tradescope.optimize.backtesting.generate_backtest_stats')

    tradeai_conf['candle_type_def'] = CandleType.FUTURES
    tradeai_conf.get('exchange', {}).update({'pair_whitelist': ['XRP/USDT:USDT']})
    tradeai_conf.get('tradeai', {}).get('feature_parameters', {}).update(
        {'include_timeframes': ['5m', '1h'], 'include_corr_pairlist': []})
    tradeai_conf['timerange'] = '20211120-20211121'

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)

    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)

    timerange = TimeRange.parse_timerange("20211115-20211122")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)

    backtesting = Backtesting(deepcopy(tradeai_conf))
    backtesting.start()

    gbs.call_args[1]['min_date'] == datetime(2021, 11, 20, 0, 0, tzinfo=timezone.utc)
    gbs.call_args[1]['max_date'] == datetime(2021, 11, 21, 0, 0, tzinfo=timezone.utc)
    Backtesting.cleanup()

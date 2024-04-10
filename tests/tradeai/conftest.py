import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from tests.conftest import get_patched_exchange
from tradescope.configuration import TimeRange
from tradescope.data.dataprovider import DataProvider
from tradescope.resolvers import StrategyResolver
from tradescope.resolvers.tradeaimodel_resolver import TradeaiModelResolver
from tradescope.tradeai.data_drawer import TradeaiDataDrawer
from tradescope.tradeai.data_kitchen import TradeaiDataKitchen


def is_py12() -> bool:
    return sys.version_info >= (3, 12)


def is_mac() -> bool:
    machine = platform.system()
    return "Darwin" in machine


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


@pytest.fixture(autouse=True)
def patch_torch_initlogs(mocker) -> None:

    if is_mac():
        # Mock torch import completely
        import sys
        import types

        module_name = 'torch'
        mocked_module = types.ModuleType(module_name)
        sys.modules[module_name] = mocked_module
    else:
        mocker.patch("torch._logging._init_logs")


@pytest.fixture(scope="function")
def tradeai_conf(default_conf, tmp_path):
    tradeaiconf = deepcopy(default_conf)
    tradeaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "strategy": "tradeai_test_strat",
            "user_data_dir": tmp_path,
            "strategy-path": "tradescope/tests/strategy/strats",
            "tradeaimodel": "LightGBMRegressor",
            "tradeaimodel_path": "tradeai/prediction_models",
            "timerange": "20180110-20180115",
            "tradeai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "unique-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path('config_examples', 'config_tradeai.example.json')]
        }
    )
    tradeaiconf['exchange'].update({'pair_whitelist': ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC']})
    return tradeaiconf


def make_rl_config(conf):
    conf.update({"strategy": "tradeai_rl_test_strat"})
    conf["tradeai"].update({"model_training_parameters": {
        "learning_rate": 0.00025,
        "gamma": 0.9,
        "verbose": 1
    }})
    conf["tradeai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {
            "rr": 1,
            "profit_aim": 0.02,
            "win_reward_factor": 2
        },
        "drop_ohlc_from_features": False
        }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> Dict[str, Any]:
    return {
            "learning_rate": 3e-4,
            "trainer_kwargs": {
                "n_steps": None,
                "batch_size": 64,
                "n_epochs": 1,
            },
            "model_kwargs": {
                "hidden_dim": 32,
                "dropout_percent": 0.2,
                "n_layer": 1,
            }
        }


def get_patched_data_kitchen(mocker, tradeaiconf):
    dk = TradeaiDataKitchen(tradeaiconf)
    return dk


def get_patched_data_drawer(mocker, tradeaiconf):
    # dd = mocker.patch('tradescope.tradeai.data_drawer', MagicMock())
    dd = TradeaiDataDrawer(tradeaiconf)
    return dd


def get_patched_tradeai_strategy(mocker, tradeaiconf):
    strategy = StrategyResolver.load_strategy(tradeaiconf)
    strategy.ts_bot_start()

    return strategy


def get_patched_tradeaimodel(mocker, tradeaiconf):
    tradeaimodel = TradeaiModelResolver.load_tradeaimodel(tradeaiconf)

    return tradeaimodel


def make_unfiltered_dataframe(mocker, tradeai_conf):
    tradeai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    tradeai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(data_load_timerange, tradeai.dk)

    tradeai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = tradeai.dd.get_base_and_corr_dataframes(
            data_load_timerange, tradeai.dk.pair, tradeai.dk
        )

    unfiltered_dataframe = tradeai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, tradeai.dk.pair
            )
    for i in range(5):
        unfiltered_dataframe[f'constant_{i}'] = i

    unfiltered_dataframe = tradeai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return tradeai, unfiltered_dataframe


def make_data_dictionary(mocker, tradeai_conf):
    tradeai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    tradeai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(data_load_timerange, tradeai.dk)

    tradeai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = tradeai.dd.get_base_and_corr_dataframes(
            data_load_timerange, tradeai.dk.pair, tradeai.dk
        )

    unfiltered_dataframe = tradeai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, tradeai.dk.pair
            )

    unfiltered_dataframe = tradeai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    tradeai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = tradeai.dk.filter_features(
            unfiltered_dataframe,
            tradeai.dk.training_features_list,
            tradeai.dk.label_list,
            training_filter=True,
        )

    data_dictionary = tradeai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = tradeai.dk.normalize_data(data_dictionary)

    return tradeai


def get_tradeai_live_analyzed_dataframe(mocker, tradeaiconf):
    strategy = get_patched_tradeai_strategy(mocker, tradeaiconf)
    exchange = get_patched_exchange(mocker, tradeaiconf)
    strategy.dp = DataProvider(tradeaiconf, exchange)
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeaiconf, tradeai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradeai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair('ADA/BTC', '5m')
    return strategy.dp.get_analyzed_dataframe('ADA/BTC', '5m')


def get_tradeai_analyzed_dataframe(mocker, tradeaiconf):
    strategy = get_patched_tradeai_strategy(mocker, tradeaiconf)
    exchange = get_patched_exchange(mocker, tradeaiconf)
    strategy.dp = DataProvider(tradeaiconf, exchange)
    strategy.tradeai_info = tradeaiconf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeaiconf, tradeai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradeai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradeai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return tradeai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')


def get_ready_to_train(mocker, tradeaiconf):
    strategy = get_patched_tradeai_strategy(mocker, tradeaiconf)
    exchange = get_patched_exchange(mocker, tradeaiconf)
    strategy.dp = DataProvider(tradeaiconf, exchange)
    strategy.tradeai_info = tradeaiconf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeaiconf, tradeai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradeai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradeai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, tradeai, strategy

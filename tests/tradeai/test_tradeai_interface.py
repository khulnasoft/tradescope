import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, log_has_re
from tests.tradeai.conftest import (get_patched_tradeai_strategy, is_arm, is_mac, make_rl_config,
                                    mock_pytorch_mlp_model_training_parameters)
from tradescope.configuration import TimeRange
from tradescope.data.dataprovider import DataProvider
from tradescope.enums import RunMode
from tradescope.optimize.backtesting import Backtesting
from tradescope.persistence import Trade
from tradescope.plugins.pairlistmanager import PairListManager
from tradescope.tradeai.data_kitchen import TradeaiDataKitchen
from tradescope.tradeai.utils import download_all_data_for_training, get_required_data_timerange


def can_run_model(model: str) -> None:
    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model

    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    if is_pytorch_model and is_mac():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    ('LightGBMRegressor', True, False, True, True, False, 0, 0),
    ('XGBoostRegressor', False, True, False, True, False, 10, 0.05),
    ('XGBoostRFRegressor', False, False, False, True, False, 0, 0),
    ('CatboostRegressor', False, False, False, True, True, 0, 0),
    ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0),
    ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner', False, True, False, True, False, 0, 0),
    ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0),
    ])
def test_extract_data_and_train_model_Standard(mocker, tradeai_conf, model, pca,
                                               dbscan, float32, can_short, shuffle,
                                               buffer, noise):

    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = 'joblib'
    tradeai_conf.update({"tradeaimodel": model})
    tradeai_conf.update({"timerange": "20180110-20180130"})
    tradeai_conf.update({"strategy": "tradeai_test_strat"})
    tradeai_conf['tradeai']['feature_parameters'].update({"principal_component_analysis": pca})
    tradeai_conf['tradeai']['feature_parameters'].update({"use_DBSCAN_to_remove_outliers": dbscan})
    tradeai_conf.update({"reduce_df_footprint": float32})
    tradeai_conf['tradeai']['feature_parameters'].update({"shuffle_after_split": shuffle})
    tradeai_conf['tradeai']['feature_parameters'].update({"buffer_train_data_candles": buffer})
    tradeai_conf['tradeai']['feature_parameters'].update({"noise_standard_deviation": noise})

    if 'ReinforcementLearner' in model:
        model_save_ext = 'zip'
        tradeai_conf = make_rl_config(tradeai_conf)
        # test the RL guardrails
        tradeai_conf['tradeai']['feature_parameters'].update({"use_SVM_to_remove_outliers": True})
        tradeai_conf['tradeai']['feature_parameters'].update({"DI_threshold": 2})
        tradeai_conf['tradeai']['data_split_parameters'].update({'shuffle': True})

    if 'test_3ac' in model or 'test_4ac' in model:
        tradeai_conf["tradeaimodel_path"] = str(Path(__file__).parents[1] / "tradeai" / "test_models")
        tradeai_conf["tradeai"]["rl_config"]["drop_ohlc_from_features"] = True

    if 'PyTorch' in model:
        model_save_ext = 'zip'
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradeai_conf['tradeai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            tradeai_conf.update({"conv_width": 10})

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.activate_tensorboard = test_tb
    tradeai.can_short = can_short
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    tradeai.dk.set_paths('ADA/BTC', 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)

    tradeai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    tradeai.dk.set_paths('ADA/BTC', None)

    tradeai.train_timer("start", "ADA/BTC")
    tradeai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradeai.dk, data_load_timerange)
    tradeai.train_timer("stop", "ADA/BTC")
    tradeai.dd.save_metric_tracker_to_disk()
    tradeai.dd.save_drawer_to_disk()

    assert Path(tradeai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(tradeai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(tradeai.dk.data_path /
                f"{tradeai.dk.model_filename}_model.{model_save_ext}").is_file()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_metadata.json").is_file()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(tradeai.dk.full_path))


@pytest.mark.parametrize('model, strat', [
    ('LightGBMRegressorMultiTarget', "tradeai_test_multimodel_strat"),
    ('XGBoostRegressorMultiTarget', "tradeai_test_multimodel_strat"),
    ('CatboostRegressorMultiTarget', "tradeai_test_multimodel_strat"),
    ('LightGBMClassifierMultiTarget', "tradeai_test_multimodel_classifier_strat"),
    ('CatboostClassifierMultiTarget', "tradeai_test_multimodel_classifier_strat")
    ])
def test_extract_data_and_train_model_MultiTargets(mocker, tradeai_conf, model, strat):
    can_run_model(model)

    tradeai_conf.update({"timerange": "20180110-20180130"})
    tradeai_conf.update({"strategy": strat})
    tradeai_conf.update({"tradeaimodel": model})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)

    tradeai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradeai.dk.set_paths('ADA/BTC', None)

    tradeai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradeai.dk, data_load_timerange)

    assert len(tradeai.dk.label_list) == 2
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_model.joblib").is_file()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_metadata.json").is_file()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(tradeai.dk.data['training_features_list']) == 14

    shutil.rmtree(Path(tradeai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    'XGBoostRFClassifier',
    'SKLearnRandomForestClassifier',
    'PyTorchMLPClassifier',
    ])
def test_extract_data_and_train_model_Classifiers(mocker, tradeai_conf, model):
    can_run_model(model)

    tradeai_conf.update({"tradeaimodel": model})
    tradeai_conf.update({"strategy": "tradeai_test_classifier"})
    tradeai_conf.update({"timerange": "20180110-20180130"})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)

    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)

    tradeai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradeai.dk.set_paths('ADA/BTC', None)

    tradeai.extract_data_and_train_model(new_timerange, "ADA/BTC",
                                        strategy, tradeai.dk, data_load_timerange)

    if 'PyTorchMLPClassifier':
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradeai_conf['tradeai']['model_training_parameters'].update(pytorch_mlp_mtp)

    if tradeai.dd.model_type == 'joblib':
        model_file_extension = ".joblib"
    elif tradeai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(f"Unsupported model type: {tradeai.dd.model_type},"
                        f" can't assign model_file_extension")

    assert Path(tradeai.dk.data_path /
                f"{tradeai.dk.model_filename}_model{model_file_extension}").exists()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_metadata.json").exists()
    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(tradeai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "tradeai_test_strat"),
        ("XGBoostRegressor", 2, "tradeai_test_strat"),
        ("CatboostRegressor", 2, "tradeai_test_strat"),
        ("PyTorchMLPRegressor", 2, "tradeai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "tradeai_test_strat"),
        ("ReinforcementLearner", 3, "tradeai_rl_test_strat"),
        ("XGBoostClassifier", 2, "tradeai_test_classifier"),
        ("LightGBMClassifier", 2, "tradeai_test_classifier"),
        ("CatboostClassifier", 2, "tradeai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "tradeai_test_classifier")
    ],
    )
def test_start_backtesting(mocker, tradeai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac() and not is_arm():
        test_tb = False

    tradeai_conf.get("tradeai", {}).update({"save_backtest_models": True})
    tradeai_conf['runmode'] = RunMode.BACKTEST

    Trade.use_db = False

    tradeai_conf.update({"tradeaimodel": model})
    tradeai_conf.update({"timerange": "20180120-20180130"})
    tradeai_conf.update({"strategy": strat})

    if 'ReinforcementLearner' in model:
        tradeai_conf = make_rl_config(tradeai_conf)

    if 'test_4ac' in model:
        tradeai_conf["tradeaimodel_path"] = str(Path(__file__).parents[1] / "tradeai" / "test_models")

    if 'PyTorch' in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradeai_conf['tradeai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            tradeai_conf.update({"conv_width": 10})

    tradeai_conf.get("tradeai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False
    tradeai.activate_tensorboard = test_tb
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradeai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradeai.dk)
    df = base_df[tradeai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    tradeai.dk.set_paths('LTC/BTC', None)
    tradeai.start_backtesting(df, metadata, tradeai.dk, strategy)
    model_folders = [x for x in tradeai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(tradeai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, tradeai_conf):
    tradeai_conf.update({"timerange": "20180120-20180124"})
    tradeai_conf['runmode'] = 'backtest'
    tradeai_conf.get("tradeai", {}).update({
        "backtest_period_days": 0.5,
        "save_backtest_models": True,
    })
    tradeai_conf.get("tradeai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradeai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradeai.dk)
    df = base_df[tradeai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    tradeai.start_backtesting(df, metadata, tradeai.dk, strategy)
    model_folders = [x for x in tradeai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(tradeai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, tradeai_conf, caplog):
    tradeai_conf.update({"timerange": "20180120-20180130"})
    tradeai_conf['runmode'] = 'backtest'
    tradeai_conf.get("tradeai", {}).update({"save_backtest_models": True})
    tradeai_conf.get("tradeai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = tradeai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradeai.dk)
    df = base_df[tradeai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    tradeai.dk.pair = pair
    tradeai.start_backtesting(df, metadata, tradeai.dk, strategy)
    model_folders = [x for x in tradeai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    tradeai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradeai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradeai.dk)
    df = base_df[tradeai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    tradeai.dk.pair = pair
    tradeai.start_backtesting(df, metadata, tradeai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    tradeai.dk.pair = pair
    tradeai.start_backtesting(df, metadata, tradeai.dk, strategy)

    path = (tradeai.dd.full_path / tradeai.dk.backtest_predictions_folder)
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(tradeai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, tradeai_conf, caplog):
    tradeai_conf['runmode'] = 'backtest'
    tradeai_conf.get("tradeai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = tradeai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradeai.dk)
    df = tradeai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_tradeai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = tradeai.dk.remove_special_chars_from_feature_names(df)
    tradeai.dk.get_unique_classes_from_labels(df)
    tradeai.dk.pair = "ADA/BTC"
    tradeai.dk.full_df = df.fillna(0)
    tradeai.dk.full_df
    assert "&-s_close_mean" not in tradeai.dk.full_df.columns
    assert "&-s_close_std" not in tradeai.dk.full_df.columns
    tradeai.backtesting_fit_live_predictions(tradeai.dk)
    assert "&-s_close_mean" in tradeai.dk.full_df.columns
    assert "&-s_close_std" in tradeai.dk.full_df.columns
    shutil.rmtree(Path(tradeai.dk.full_path))


def test_plot_feature_importance(mocker, tradeai_conf):

    from tradescope.tradeai.utils import plot_feature_importance

    tradeai_conf.update({"timerange": "20180110-20180130"})
    tradeai_conf.get("tradeai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"})

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = True
    tradeai.dk = TradeaiDataKitchen(tradeai_conf)
    tradeai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradeai.dd.load_all_pair_histories(timerange, tradeai.dk)

    tradeai.dd.pair_dict = {"ADA/BTC": {"model_filename": "fake_name",
                                       "trained_timestamp": 1, "data_path": "", "extras": {}}}

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradeai.dk.set_paths('ADA/BTC', None)

    tradeai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradeai.dk, data_load_timerange)

    model = tradeai.dd.load_data("ADA/BTC", tradeai.dk)

    plot_feature_importance(model, "ADA/BTC", tradeai.dk)

    assert Path(tradeai.dk.data_path / f"{tradeai.dk.model_filename}.html")

    shutil.rmtree(Path(tradeai.dk.full_path))


@pytest.mark.parametrize('timeframes,corr_pairs', [
    (['5m'], ['ADA/BTC', 'DASH/BTC']),
    (['5m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
    (['5m', '15m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
])
def test_tradeai_informative_pairs(mocker, tradeai_conf, timeframes, corr_pairs):
    tradeai_conf['tradeai']['feature_parameters'].update({
        'include_timeframes': timeframes,
        'include_corr_pairlist': corr_pairs,

    })
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    pairlists = PairListManager(exchange, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, tradeai_conf, caplog):
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    pairlist = PairListManager(exchange, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange, pairlist)
    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.live = False

    tradeai.train_queue = tradeai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, tradeai_conf):
    time_range = get_required_data_timerange(tradeai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, tradeai_conf, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    pairlist = PairListManager(exchange, tradeai_conf)
    strategy.dp = DataProvider(tradeai_conf, exchange, pairlist)
    tradeai_conf['pairs'] = tradeai_conf['exchange']['pair_whitelist']
    tradeai_conf['datadir'] = tmp_path
    download_all_data_for_training(strategy.dp, tradeai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('dp_exists', [(False), (True)])
def test_get_state_info(mocker, tradeai_conf, dp_exists, caplog, tickers):

    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")

    tradeai_conf.update({"tradeaimodel": "ReinforcementLearner"})
    tradeai_conf.update({"timerange": "20180110-20180130"})
    tradeai_conf.update({"strategy": "tradeai_rl_test_strat"})
    tradeai_conf = make_rl_config(tradeai_conf)
    tradeai_conf['entry_pricing']['price_side'] = 'same'
    tradeai_conf['exit_pricing']['price_side'] = 'same'

    strategy = get_patched_tradeai_strategy(mocker, tradeai_conf)
    exchange = get_patched_exchange(mocker, tradeai_conf)
    ticker_mock = MagicMock(return_value=tickers()['ETH/BTC'])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(tradeai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.tradeai_info = tradeai_conf.get("tradeai", {})
    tradeai = strategy.tradeai
    tradeai.data_provider = strategy.dp
    tradeai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    tradeai.get_state_info("ADA/BTC")
    tradeai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )

# pragma pylint: disable=missing-docstring

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from tradescope.commands import Arguments
from tradescope.enums import State
from tradescope.exceptions import ConfigurationError, TradescopeException, OperationalException
from tradescope.tradescopebot import TradescopeBot
from tradescope.main import main
from tradescope.worker import Worker
from tests.conftest import (log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


def test_parse_args_None(caplog) -> None:
    with pytest.raises(SystemExit):
        main([])
    assert log_has_re(r"Usage of Tradescope requires a subcommand.*", caplog)


def test_parse_args_backtesting(mocker) -> None:
    """
    Test that main() can start backtesting and also ensure we can pass some specific arguments
    further argument parsing is done in test_arguments.py
    """
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    backtesting_mock = mocker.patch('tradescope.commands.start_backtesting')
    backtesting_mock.__name__ = PropertyMock("start_backtesting")
    # it's sys.exit(0) at the end of backtesting
    with pytest.raises(SystemExit):
        main(['backtesting'])
    assert backtesting_mock.call_count == 1
    call_args = backtesting_mock.call_args[0][0]
    assert call_args['config'] == ['config.json']
    assert call_args['verbosity'] == 0
    assert call_args['command'] == 'backtesting'
    assert call_args['func'] is not None
    assert callable(call_args['func'])
    assert call_args['timeframe'] is None


def test_main_start_hyperopt(mocker) -> None:
    mocker.patch.object(Path, 'is_file', MagicMock(side_effect=[False, True]))
    hyperopt_mock = mocker.patch('tradescope.commands.start_hyperopt', MagicMock())
    hyperopt_mock.__name__ = PropertyMock('start_hyperopt')
    # it's sys.exit(0) at the end of hyperopt
    with pytest.raises(SystemExit):
        main(['hyperopt'])
    assert hyperopt_mock.call_count == 1
    call_args = hyperopt_mock.call_args[0][0]
    assert call_args['config'] == ['config.json']
    assert call_args['verbosity'] == 0
    assert call_args['command'] == 'hyperopt'
    assert call_args['func'] is not None
    assert callable(call_args['func'])


def test_main_fatal_exception(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch('tradescope.tradescopebot.TradescopeBot.cleanup', MagicMock())
    mocker.patch('tradescope.worker.Worker._worker', MagicMock(side_effect=Exception))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('tradescope.tradescopebot.RPCManager', MagicMock())
    mocker.patch('tradescope.tradescopebot.init_db', MagicMock())

    args = ['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: tests/testdata/testconfigs/main_test_config.json ...', caplog)
    assert log_has('Fatal exception!', caplog)


def test_main_keyboard_interrupt(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch('tradescope.tradescopebot.TradescopeBot.cleanup', MagicMock())
    mocker.patch('tradescope.worker.Worker._worker', MagicMock(side_effect=KeyboardInterrupt))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('tradescope.tradescopebot.RPCManager', MagicMock())
    mocker.patch('tradescope.wallets.Wallets.update', MagicMock())
    mocker.patch('tradescope.tradescopebot.init_db', MagicMock())

    args = ['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: tests/testdata/testconfigs/main_test_config.json ...', caplog)
    assert log_has('SIGINT received, aborting ...', caplog)


def test_main_operational_exception(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch('tradescope.tradescopebot.TradescopeBot.cleanup', MagicMock())
    mocker.patch(
        'tradescope.worker.Worker._worker',
        MagicMock(side_effect=TradescopeException('Oh snap!'))
    )
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('tradescope.wallets.Wallets.update', MagicMock())
    mocker.patch('tradescope.tradescopebot.RPCManager', MagicMock())
    mocker.patch('tradescope.tradescopebot.init_db', MagicMock())

    args = ['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: tests/testdata/testconfigs/main_test_config.json ...', caplog)
    assert log_has('Oh snap!', caplog)


def test_main_operational_exception1(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch(
        'tradescope.commands.list_commands.list_available_exchanges',
        MagicMock(side_effect=ValueError('Oh snap!'))
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ['list-exchanges']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)

    assert log_has('Fatal exception!', caplog)
    assert not log_has_re(r'SIGINT.*', caplog)
    mocker.patch(
        'tradescope.commands.list_commands.list_available_exchanges',
        MagicMock(side_effect=KeyboardInterrupt)
    )
    with pytest.raises(SystemExit):
        main(args)

    assert log_has_re(r'SIGINT.*', caplog)


def test_main_ConfigurationError(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch(
        'tradescope.commands.list_commands.list_available_exchanges',
        MagicMock(side_effect=ConfigurationError('Oh snap!'))
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ['list-exchanges']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has_re('Configuration error: Oh snap!', caplog)


def test_main_reload_config(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch('tradescope.tradescopebot.TradescopeBot.cleanup', MagicMock())
    # Simulate Running, reload, running workflow
    worker_mock = MagicMock(side_effect=[State.RUNNING,
                                         State.RELOAD_CONFIG,
                                         State.RUNNING,
                                         OperationalException("Oh snap!")])
    mocker.patch('tradescope.worker.Worker._worker', worker_mock)
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('tradescope.wallets.Wallets.update', MagicMock())
    reconfigure_mock = mocker.patch('tradescope.worker.Worker._reconfigure', MagicMock())

    mocker.patch('tradescope.tradescopebot.RPCManager', MagicMock())
    mocker.patch('tradescope.tradescopebot.init_db', MagicMock())

    args = Arguments([
        'trade',
        '-c',
        'tests/testdata/testconfigs/main_test_config.json'
    ]).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    with pytest.raises(SystemExit):
        main(['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json'])

    assert log_has('Using config: tests/testdata/testconfigs/main_test_config.json ...', caplog)
    assert worker_mock.call_count == 4
    assert reconfigure_mock.call_count == 1
    assert isinstance(worker.tradescope, TradescopeBot)


def test_reconfigure(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('tradescope.tradescopebot.TradescopeBot.cleanup', MagicMock())
    mocker.patch(
        'tradescope.worker.Worker._worker',
        MagicMock(side_effect=OperationalException('Oh snap!'))
    )
    mocker.patch('tradescope.wallets.Wallets.update', MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('tradescope.tradescopebot.RPCManager', MagicMock())
    mocker.patch('tradescope.tradescopebot.init_db', MagicMock())

    args = Arguments([
        'trade',
        '-c',
        'tests/testdata/testconfigs/main_test_config.json'
    ]).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    tradescope = worker.tradescope

    # Renew mock to return modified data
    conf = deepcopy(default_conf)
    conf['stake_amount'] += 1
    patched_configuration_load_config_file(mocker, conf)

    worker._config = conf
    # reconfigure should return a new instance
    worker._reconfigure()
    tradescope2 = worker.tradescope

    # Verify we have a new instance with the new config
    assert tradescope is not tradescope2
    assert tradescope.config['stake_amount'] + 1 == tradescope2.config['stake_amount']

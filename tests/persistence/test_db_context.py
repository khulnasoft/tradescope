import pytest

from tradescope.persistence import PairLocks, Trade, TsNoDBContext


@pytest.mark.parametrize('timeframe', ['', '5m', '1d'])
def test_TsNoDBContext(timeframe):
    PairLocks.timeframe = ''
    assert Trade.use_db is True
    assert PairLocks.use_db is True
    assert PairLocks.timeframe == ''

    with TsNoDBContext(timeframe):
        assert Trade.use_db is False
        assert PairLocks.use_db is False
        assert PairLocks.timeframe == timeframe

    with TsNoDBContext():
        assert Trade.use_db is False
        assert PairLocks.use_db is False
        assert PairLocks.timeframe == ''

    assert Trade.use_db is True
    assert PairLocks.use_db is True

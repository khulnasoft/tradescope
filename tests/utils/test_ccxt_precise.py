from tradescope.util import TsPrecise


ws = TsPrecise('-1.123e-6')
ws = TsPrecise('-1.123e-6')
xs = TsPrecise('0.00000002')
ys = TsPrecise('69696900000')
zs = TsPrecise('0')


def test_TsPrecise():
    assert ys * xs == '1393.938'
    assert xs * ys == '1393.938'

    assert ys + xs == '69696900000.00000002'
    assert xs + ys == '69696900000.00000002'
    assert xs - ys == '-69696899999.99999998'
    assert ys - xs == '69696899999.99999998'
    assert xs / ys == '0'
    assert ys / xs == '3484845000000000000'

    assert ws * xs == '-0.00000000000002246'
    assert xs * ws == '-0.00000000000002246'

    assert ws + xs == '-0.000001103'
    assert xs + ws == '-0.000001103'

    assert xs - ws == '0.000001143'
    assert ws - xs == '-0.000001143'

    assert xs / ws == '-0.017809439002671415'
    assert ws / xs == '-56.15'

    assert zs * ws == '0'
    assert zs * xs == '0'
    assert zs * ys == '0'
    assert ws * zs == '0'
    assert xs * zs == '0'
    assert ys * zs == '0'

    assert zs + ws == '-0.000001123'
    assert zs + xs == '0.00000002'
    assert zs + ys == '69696900000'
    assert ws + zs == '-0.000001123'
    assert xs + zs == '0.00000002'
    assert ys + zs == '69696900000'

    assert abs(TsPrecise('-500.1')) == '500.1'
    assert abs(TsPrecise('213')) == '213'

    assert abs(TsPrecise('-500.1')) == '500.1'
    assert -TsPrecise('213') == '-213'

    assert TsPrecise('10.1') % TsPrecise('0.5') == '0.1'
    assert TsPrecise('5550') % TsPrecise('120') == '30'

    assert TsPrecise('-0.0') == TsPrecise('0')
    assert TsPrecise('5.534000') == TsPrecise('5.5340')

    assert min(TsPrecise('-3.1415'), TsPrecise('-2')) == '-3.1415'

    assert max(TsPrecise('3.1415'), TsPrecise('-2')) == '3.1415'

    assert TsPrecise('2') > TsPrecise('1.2345')
    assert not TsPrecise('-3.1415') > TsPrecise('-2')
    assert not TsPrecise('3.1415') > TsPrecise('3.1415')
    assert TsPrecise.string_gt('3.14150000000000000000001', '3.1415')

    assert TsPrecise('3.1415') >= TsPrecise('3.1415')
    assert TsPrecise('3.14150000000000000000001') >= TsPrecise('3.1415')

    assert not TsPrecise('3.1415') < TsPrecise('3.1415')

    assert TsPrecise('3.1415') <= TsPrecise('3.1415')
    assert TsPrecise('3.1415') <= TsPrecise('3.14150000000000000000001')

    assert TsPrecise(213) == '213'
    assert TsPrecise(-213) == '-213'
    assert str(TsPrecise(-213)) == '-213'
    assert TsPrecise(213.2) == '213.2'
    assert float(TsPrecise(213.2)) == 213.2
    assert float(TsPrecise(-213.2)) == -213.2

from minerva.strategy import btc_params as bp


def test_recent_counts_window_bars():
    assert bp.recent([True, False, False, False], 3) == [True, True, True, False]
    assert bp.recent([True, False], 1) == [True, False]


def test_levels_stop_at_first_missing_indicator():
    a = [True, False, True]
    b = [True, False, False]
    c = [True, True, True]
    assert bp.levels([a, b, c], 1) == [3, 0, 1]


def test_levels_window_joins_nearby_bars():
    a = [True, False, False]
    b = [False, True, False]
    assert bp.levels([a, b], 2) == [1, 2, 0]


def test_crosses_detect_both_sides():
    up, down = bp.crosses([10, 30, 90, 70], 20, 80)
    assert up == [False, True, False, False] and down == [False, False, False, True]


def test_pick_uses_only_earlier_years_and_goes_flat():
    good = {
        2019: {"final": 1.5, "mdd": -0.1, "trades": 40},
        2020: {"final": 0.1, "mdd": -1, "trades": 40},
    }
    late = {
        2019: {"final": 0.9, "mdd": -0.1, "trades": 40},
        2020: {"final": 9.0, "mdd": 0, "trades": 40},
    }
    assert bp.pick({"a": good, "b": late}, 2020) == "a"
    assert bp.pick({"b": late}, 2020) is None
    few = {2019: {"final": 2.0, "mdd": 0, "trades": 5}}
    assert bp.pick({"c": few}, 2020) is None


def test_walk_compounds_chosen_years_only():
    r2019 = {"final": 1.5, "mdd": -0.1, "trades": 40}
    table = {"a": {2019: r2019, **{y: {"final": 1.1, "mdd": -0.2, "trades": 40} for y in bp.WALK}}}
    got = bp.walk(table, table)
    assert abs(got["final"] - 1.1 ** len(bp.WALK)) < 1e-9 and got["worst_year_mdd"] == -0.2

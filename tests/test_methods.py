from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dss.methods import (ahp_scores, ahp_weights, electre_i, matrix_from_upper,
                         pairwise_from_points, pareto_front, rank, read_alternatives,
                         run_all, spearman, topsis)

DATA = Path(__file__).resolve().parent.parent / "data"


def test_ahp_matches_teacher_example_4x4():
    # Этап 1 из файла "Метод Саати.xls", лист МАИ
    M = [[1, 1, 3, 5], [1, 1, 3, 5], [1 / 3, 1 / 3, 1, 3], [1 / 5, 1 / 5, 1 / 3, 1]]
    w, lam, ci, cr = ahp_weights(M)
    assert np.allclose(w, [0.390813, 0.390813, 0.150892, 0.067481], atol=1e-6)
    assert lam == pytest.approx(4.031397, abs=1e-6)
    assert ci == pytest.approx(0.010466, abs=1e-6)
    assert cr == pytest.approx(0.011629, abs=1e-6)


def test_ahp_matches_teacher_example_9x9_inconsistent():
    # Лист "Пример": там T = 0,216 > 0,1, матрица заполнена некорректно
    upper = [[1, 5, 7, 7, 5, 1, 3, 5, 1],
             [0, 1, 5, 9, 3, 1 / 7, 1 / 5, 1 / 3, 1 / 5],
             [0, 0, 1, 5, 1 / 5, 1 / 9, 1 / 5, 1 / 7, 1 / 9],
             [0, 0, 0, 1, 1 / 3, 1 / 9, 1 / 7, 1 / 5, 1 / 9],
             [0, 0, 0, 0, 1, 3, 5, 3, 1 / 5],
             [0, 0, 0, 0, 0, 1, 5, 7, 3],
             [0, 0, 0, 0, 0, 0, 1, 5, 1 / 5],
             [0, 0, 0, 0, 0, 0, 0, 1, 1 / 7],
             [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    M = matrix_from_upper(9, {(i, j): upper[i][j] for i in range(9) for j in range(i + 1, 9)})
    _, lam, _, cr = ahp_weights(M)
    assert lam == pytest.approx(11.507858, abs=1e-5)
    assert cr == pytest.approx(0.216195, abs=1e-5)
    assert cr > 0.1


def test_pairwise_from_points_is_reciprocal():
    M = pairwise_from_points([5, 4, 2, 1])
    assert np.allclose(M * M.T, 1)
    assert M[0, 1] == 3 and M[0, 2] == 7 and M[0, 3] == 9


def test_ahp_scores_directions_and_sum():
    X = np.array([[10, 5], [20, 5]], dtype=float)
    s, P = ahp_scores(X, ["min", "max"], [0.5, 0.5])
    assert s.sum() == pytest.approx(1)
    assert s[0] > s[1]                      # дешевле при равном втором критерии
    assert P[:, 0] == pytest.approx([2 / 3, 1 / 3])


def test_topsis_dominant_is_ideal():
    X = np.array([[10, 9], [20, 3]], dtype=float)
    c, dp, dm = topsis(X, ["min", "max"], [0.5, 0.5])
    assert c[0] == pytest.approx(1) and c[1] == pytest.approx(0)


def test_topsis_hand_example():
    X = np.array([[3, 4], [4, 3]], dtype=float)
    c, _, _ = topsis(X, ["max", "max"], [0.7, 0.3])
    # первый критерий важнее, по нему лучше вторая альтернатива
    assert c[1] > c[0]
    assert c[0] + c[1] == pytest.approx(1)


def test_electre_dominance_and_thresholds():
    X = np.array([[10, 9, 1], [20, 3, 2], [15, 6, 1]], dtype=float)
    dirs = ["min", "max", "min"]
    net, out, C, D, kernel = electre_i(X, dirs, [1, 1, 1], 0.6, 0.4)
    assert out[0, 1] and out[0, 2] and not out[1, 0]
    assert C[0, 1] == pytest.approx(1) and D[0, 1] == 0
    assert kernel.tolist() == [True, False, False]
    assert rank(net)[0] == 1


def test_pareto():
    X = np.array([[10, 9], [20, 3], [5, 2], [12, 9]], dtype=float)
    assert pareto_front(X, ["min", "max"]).tolist() == [True, False, True, False]


def test_read_alternatives_parses_directions():
    df = pd.DataFrame({"Имя": ["A", "B"], "Цена (min)": [1, 2], "Качество (max)": [3, 4], "Вес": [1, 1]})
    names, X, crit, dirs = read_alternatives(df)
    assert crit == ["Цена", "Качество", "Вес"] and dirs == ["min", "max", "max"]


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"Имя": ["A", "B"], "Цена": [1, "x"], "К": [1, 2]}),
    pd.DataFrame({"Имя": ["A", "B"], "Цена": [0, 2], "К": [1, 2]}),
    pd.DataFrame({"Имя": ["A"], "Цена": [1], "К": [1]}),
])
def test_read_alternatives_rejects_bad(bad):
    with pytest.raises(ValueError):
        read_alternatives(bad)


@pytest.mark.parametrize("path", sorted(DATA.glob("variant_*.csv")))
def test_all_demo_variants_run(path):
    names, X, crit, dirs = read_alternatives(pd.read_csv(path))
    assert X.shape == (5, 9)
    w, *_ = ahp_weights(np.ones((9, 9)))
    table, *_ = run_all(names, X, dirs, w)
    assert table["AHP"].sum() == pytest.approx(1)
    assert table["Парето"].any()


def test_spearman_identical():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1)

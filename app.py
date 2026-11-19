"""Система поддержки принятия решений: AHP (Саати), TOPSIS, ELECTRE I и Парето.
Запуск: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dss.methods import (DEFAULT_POINTS, ahp_weights, load_variants, matrix_from_upper,
                         pairwise_from_points, read_alternatives, read_csv_any, run_all, spearman)

DATA_FILE = Path(__file__).parent / "data" / "variants.csv"

# Шкала Саати для попарных сравнений: подпись -> значение a_ij
SCALE = {
    "A намного важнее (9)": 9, "A сильно важнее (7)": 7, "A заметно важнее (5)": 5,
    "A немного важнее (3)": 3, "Одинаково (1)": 1, "B немного важнее (1/3)": 1 / 3,
    "B заметно важнее (1/5)": 1 / 5, "B сильно важнее (1/7)": 1 / 7, "B намного важнее (1/9)": 1 / 9,
}
SCALE_LABELS = list(SCALE.keys())


def nearest_label(value):
    return min(SCALE_LABELS, key=lambda k: abs(np.log(SCALE[k]) - np.log(value)))


st.set_page_config(page_title="СППР: AHP, TOPSIS, ELECTRE", layout="wide")
st.title("Система поддержки принятия решений")
st.caption("Сравнение альтернатив методами AHP (Саати), TOPSIS и ELECTRE I, проверка Парето")

# ---------- 1. Данные ----------
with st.sidebar:
    st.header("1. Данные")
    source = st.radio("Источник", ["Пример из курса", "Свой CSV"])
    if source == "Пример из курса":
        variants = load_variants(DATA_FILE)
        variant = st.selectbox("Вариант (бизнес-проекты)", sorted(variants),
                               format_func=lambda v: f"Вариант {v}")
        raw = variants[variant]
    else:
        up = st.file_uploader("CSV: первый столбец - названия, далее числовые критерии", type="csv")
        st.caption('Направление критерия можно указать в названии столбца: "Цена (min)", "Качество (max)".')
        if up is None:
            st.info("Загрузи файл или выбери пример.")
            st.stop()
        try:
            raw = read_csv_any(up.getvalue())
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")
            st.stop()
    template = load_variants(DATA_FILE)[1].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать шаблон CSV", template, "template.csv", "text/csv")

try:
    names, X_all, criteria_all, dirs_all = read_alternatives(raw)
    if len(set(names)) != len(names):
        raise ValueError("Названия альтернатив в первом столбце должны быть разными.")
except ValueError as e:
    st.error(f"Ошибка в данных: {e}")
    st.stop()

tab_data, tab_w, tab_res = st.tabs(["Данные и критерии", "Веса критериев (AHP)", "Рейтинг"])

with tab_data:
    st.subheader("Альтернативы")
    st.dataframe(raw, hide_index=True)
    st.subheader("Критерии")
    chosen = st.multiselect("Какие критерии учитывать", criteria_all, default=criteria_all)
    if len(chosen) < 2:
        st.warning("Выбери минимум два критерия.")
        st.stop()
    idx = [criteria_all.index(c) for c in chosen]
    cols = st.columns(3)
    directions = []
    for k, i in enumerate(idx):
        d = cols[k % 3].selectbox(chosen[k], ["max", "min"], index=0 if dirs_all[i] == "max" else 1,
                                  format_func=lambda v: "больше = лучше" if v == "max" else "меньше = лучше",
                                  key=f"dir_{chosen[k]}")
        directions.append(d)
    X = X_all[:, idx]

# ---------- 2. Веса ----------
with tab_w:
    n = len(chosen)
    mode = st.radio("Как задать важность критериев",
                    ["Быстро: баллы от 1 до 5", "Попарные сравнения по шкале Саати"], horizontal=True)
    points = [DEFAULT_POINTS.get(c, 3) for c in chosen]
    if mode.startswith("Быстро"):
        cols = st.columns(3)
        points = [cols[k % 3].slider(c, 1, 5, points[k], key=f"pt_{c}") for k, c in enumerate(chosen)]
        M = pairwise_from_points(points)
    else:
        start = pairwise_from_points(points)
        upper = {}
        st.caption("A - критерий слева, B - критерий справа. Ниже диагонали обратные значения ставятся сами.")
        for i in range(n):
            for j in range(i + 1, n):
                lab = st.select_slider(f"{chosen[i]}  vs  {chosen[j]}", SCALE_LABELS,
                                       value=nearest_label(start[i, j]), key=f"pw_{i}_{j}")
                upper[(i, j)] = SCALE[lab]
        M = matrix_from_upper(n, upper)

    w, lam, ci, cr = ahp_weights(M)
    c1, c2, c3 = st.columns(3)
    c1.metric("λ max", f"{lam:.4f}")
    c2.metric("Индекс согласованности CI", f"{ci:.4f}")
    c3.metric("Отношение согласованности CR", f"{cr:.4f}")
    if cr < 0.1:
        st.success("CR < 0,1: суждения согласованы, веса можно использовать.")
    else:
        st.error("CR ≥ 0,1: суждения противоречивы, пересмотри сравнения.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Матрица парных сравнений**")
        st.dataframe(pd.DataFrame(M, index=chosen, columns=chosen).round(3))
    with right:
        wdf = pd.DataFrame({"Критерий": chosen, "Вес": w}).sort_values("Вес")
        st.plotly_chart(px.bar(wdf, x="Вес", y="Критерий", orientation="h", text_auto=".3f",
                               title="Веса критериев"))

# ---------- 3. Рейтинг ----------
with tab_res:
    a, b = st.columns(2)
    c_thr = a.slider("ELECTRE: порог согласия", 0.5, 1.0, 0.6, 0.05)
    d_thr = b.slider("ELECTRE: порог несогласия", 0.0, 1.0, 0.75, 0.05)
    table, out, C, D = run_all(names, X, directions, w, c_thr, d_thr)

    best = table.sort_values(["Средний ранг", "Место TOPSIS"]).iloc[0]["Альтернатива"]
    winners = {m: table.loc[table[f"Место {m}"] == 1, "Альтернатива"].tolist() for m in ["AHP", "TOPSIS", "ELECTRE"]}
    st.success(f"Рекомендация (лучший средний ранг по трём методам): **{best}**")
    st.write("Первые места: " + "; ".join(f"{m}: {', '.join(v)}" for m, v in winners.items()))
    if cr >= 0.1:
        st.warning("Веса получены из несогласованной матрицы (CR ≥ 0,1), результат ненадёжен.")

    show = table.copy()
    show["Ядро ELECTRE"] = show["Ядро ELECTRE"].map({True: "да", False: ""})
    show["Парето"] = show["Парето"].map({True: "недоминируемая", False: "доминируемая"})
    st.dataframe(show.round(4), hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        long = table.melt(id_vars="Альтернатива", value_vars=["Место AHP", "Место TOPSIS", "Место ELECTRE"],
                          var_name="Метод", value_name="Место")
        long["Метод"] = long["Метод"].str.replace("Место ", "")
        fig = px.line(long, x="Метод", y="Место", color="Альтернатива", markers=True,
                      title="Места по трём методам (1 = лучший)")
        fig.update_yaxes(autorange="reversed", dtick=1)
        st.plotly_chart(fig)
    with g2:
        method = st.selectbox("Оценки метода", ["AHP", "TOPSIS", "ELECTRE (чистый индекс)"])
        st.plotly_chart(px.bar(table.sort_values(method), x=method, y="Альтернатива", orientation="h",
                               text_auto=".3f", title=f"Оценки {method}"))

    s1, s2, s3 = st.columns(3)
    s1.metric("Спирмен AHP и TOPSIS", f"{spearman(table['AHP'], table['TOPSIS']):.2f}")
    s2.metric("Спирмен AHP и ELECTRE", f"{spearman(table['AHP'], table['ELECTRE (чистый индекс)']):.2f}")
    s3.metric("Спирмен TOPSIS и ELECTRE", f"{spearman(table['TOPSIS'], table['ELECTRE (чистый индекс)']):.2f}")

    with st.expander("ELECTRE I: матрицы согласия, несогласия и превосходства"):
        e1, e2 = st.columns(2)
        e1.plotly_chart(px.imshow(pd.DataFrame(C, index=names, columns=names), text_auto=".2f",
                                  title="Индекс согласия C(a, b)", color_continuous_scale="Blues"))
        e2.plotly_chart(px.imshow(pd.DataFrame(D, index=names, columns=names), text_auto=".2f",
                                  title="Индекс несогласия D(a, b)", color_continuous_scale="Reds"))
        rel = [f"{names[i]} превосходит {names[j]}" for i in range(len(names))
               for j in range(len(names)) if out[i, j]]
        st.write("Отношения превосходства: " + ("; ".join(rel) if rel else "нет при этих порогах"))

    st.download_button("Скачать рейтинг (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                       "rating.csv", "text/csv")

"""Методы поддержки принятия решений: AHP (Саати), TOPSIS, ELECTRE I, Парето."""
import io
import re

import numpy as np
import pandas as pd

# Показатель случайной согласованности (ПСС) из методички, n = размер матрицы
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Шкала Саати, которую использует методичка
SAATY_SCALE = [1, 3, 5, 7, 9]


# Баллы важности по умолчанию для примера с бизнес-проектами (1 = мало важен, 5 = очень важен)
DEFAULT_POINTS = {
    "Готовность": 4, "Инвестиции млн руб": 3, "Срок окупаемости мес": 2,
    "Численность команды чел": 1, "Конкурентная среда": 3, "Доход млн руб": 5,
    "Потенциальный рынок": 2, "Популярность": 2, "Риски": 5,
}


# ---------- Данные ----------

def load_variants(path):
    """Примеры из курса: один CSV, первый столбец "Вариант".
    Возвращает словарь {номер варианта: таблица альтернатив без этого столбца}."""
    df = pd.read_csv(path)
    return {int(v): g.drop(columns="Вариант").reset_index(drop=True)
            for v, g in df.groupby("Вариант")}


def read_csv_any(data: bytes) -> pd.DataFrame:
    """CSV из любого редактора: UTF-8 или Windows-1251, разделитель запятая или точка с запятой."""
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось прочитать файл: неизвестная кодировка.")
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if sep == ";":   # в русском Excel дробная часть пишется через запятую
        for c in df.columns[1:]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    return df


def read_alternatives(df: pd.DataFrame):
    """Первый столбец - названия альтернатив, остальные - числовые критерии.
    Направление критерия берётся из названия: "(min)" или "(max)", по умолчанию max.
    Возвращает: названия, матрицу X, названия критериев, список направлений."""
    if df.shape[1] < 3:
        raise ValueError("Нужен столбец с названиями и минимум два критерия.")
    if len(df) < 2:
        raise ValueError("Нужно минимум две альтернативы.")
    names = df.iloc[:, 0].astype(str).tolist()
    crit_cols = list(df.columns[1:])
    X = df[crit_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"В столбцах есть пустые или нечисловые значения: {bad}")
    X = X.to_numpy(dtype=float)
    if (X <= 0).any():
        raise ValueError("Все значения критериев должны быть больше нуля.")
    criteria, directions = [], []
    for col in crit_cols:
        m = re.search(r"\((min|max)\)\s*$", str(col), flags=re.I)
        directions.append(m.group(1).lower() if m else "max")
        criteria.append(re.sub(r"\s*\((min|max)\)\s*$", "", str(col), flags=re.I))
    return names, X, criteria, directions


# ---------- AHP (метод Саати) ----------

def ahp_weights(M):
    """Веса по матрице парных сравнений так, как в методичке:
    среднее геометрическое строк, нормировка на сумму,
    lambda = сумма (сумма столбца * вес), CI = (lambda - n)/(n - 1), CR = CI / ПСС."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    geo = np.prod(M, axis=1) ** (1.0 / n)
    w = geo / geo.sum()
    lam = float((M.sum(axis=0) * w).sum())
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return w, lam, ci, cr


def pairwise_from_points(points):
    """Матрица сравнений по баллам важности критериев (1..5).
    Разница баллов 0 -> 1, 1 -> 3, 2 -> 5, 3 -> 7, 4 и больше -> 9; обратные значения ниже диагонали."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            d = p[i] - p[j]
            v = min(1 + 2 * abs(d), 9)
            M[i, j] = v if d >= 0 else 1.0 / v
    return M


def matrix_from_upper(n, upper):
    """Собирает обратно-симметричную матрицу по значениям выше диагонали.
    upper: словарь {(i, j): значение} для i < j."""
    M = np.ones((n, n))
    for (i, j), v in upper.items():
        M[i, j] = v
        M[j, i] = 1.0 / v
    return M


def ahp_scores(X, directions, w):
    """Итоговая оценка альтернатив по AHP.
    Локальный приоритет: для max x / сумма x, для min (1/x) / сумма (1/x).
    Итог: сумма (вес критерия * локальный приоритет)."""
    X = np.asarray(X, dtype=float)
    P = np.empty_like(X)
    for j, d in enumerate(directions):
        col = X[:, j] if d == "max" else 1.0 / X[:, j]
        P[:, j] = col / col.sum()
    return P @ np.asarray(w, dtype=float), P


# ---------- TOPSIS ----------

def topsis(X, directions, w):
    """Близость к идеальному решению C = D- / (D+ + D-)."""
    X = np.asarray(X, dtype=float)
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * np.asarray(w, dtype=float)
    is_max = np.array([d == "max" for d in directions])
    best = np.where(is_max, V.max(axis=0), V.min(axis=0))
    worst = np.where(is_max, V.min(axis=0), V.max(axis=0))
    d_plus = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    c = np.divide(d_minus, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    return c, d_plus, d_minus


# ---------- ELECTRE I ----------

def electre_i(X, directions, w, c_threshold=0.6, d_threshold=0.75):
    """ELECTRE I.
    Согласие C(a,b): сумма весов критериев, где a не хуже b.
    Несогласие D(a,b): наибольший проигрыш a по критерию, делённый на размах критерия.
    a превосходит b, если C(a,b) >= порога согласия и D(a,b) <= порога несогласия.
    Для полного ранжирования используется чистый индекс:
    (сумма C(a,b) - сумма C(b,a)) - (сумма D(a,b) - сумма D(b,a)). Чем больше, тем лучше."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m, n = X.shape
    S = X.copy()
    for j, d in enumerate(directions):
        if d == "min":
            S[:, j] = -S[:, j]          # теперь по всем критериям "больше = лучше"
    span = S.max(axis=0) - S.min(axis=0)
    span[span == 0] = 1.0
    C = np.zeros((m, m))
    D = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            not_worse = S[a] >= S[b]
            C[a, b] = w[not_worse].sum()
            loss = (S[b] - S[a]) / span
            D[a, b] = max(0.0, loss.max())
    out = (C >= c_threshold - 1e-12) & (D <= d_threshold + 1e-12)
    np.fill_diagonal(out, False)
    net = (C.sum(axis=1) - C.sum(axis=0)) - (D.sum(axis=1) - D.sum(axis=0))
    kernel = ~out.any(axis=0)       # тех, кого никто не превосходит
    return net, out, C, D, kernel


# ---------- Парето ----------

def pareto_front(X, directions):
    """True для недоминируемых альтернатив."""
    X = np.asarray(X, dtype=float)
    S = X.copy()
    for j, d in enumerate(directions):
        if d == "min":
            S[:, j] = -S[:, j]
    m = len(S)
    res = np.ones(m, dtype=bool)
    for a in range(m):
        for b in range(m):
            if a != b and (S[b] >= S[a]).all() and (S[b] > S[a]).any():
                res[a] = False
                break
    return res


# ---------- Сводка ----------

def rank(scores):
    """Место: 1 = лучший, при равенстве одинаковое место."""
    return pd.Series(scores).rank(ascending=False, method="min").astype(int).to_numpy()


def spearman(r1, r2):
    a = pd.Series(r1).rank().to_numpy()
    b = pd.Series(r2).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_all(names, X, directions, w, c_threshold=0.6, d_threshold=0.75):
    """Все три метода и Парето в одной таблице."""
    ahp, _ = ahp_scores(X, directions, w)
    tps, _, _ = topsis(X, directions, w)
    net, out, C, D, kernel = electre_i(X, directions, w, c_threshold, d_threshold)
    par = pareto_front(X, directions)
    table = pd.DataFrame({
        "Альтернатива": names,
        "AHP": ahp, "Место AHP": rank(ahp),
        "TOPSIS": tps, "Место TOPSIS": rank(tps),
        "ELECTRE (чистый индекс)": net, "Место ELECTRE": rank(net),
        "Ядро ELECTRE": kernel,
        "Парето": par,
    })
    table["Средний ранг"] = table[["Место AHP", "Место TOPSIS", "Место ELECTRE"]].mean(axis=1)
    return table, out, C, D

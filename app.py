# -*- coding: utf-8 -*-
import re
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="직무테스트 자동 문항 리스크 분석",
    layout="wide",
)

st.title("직무테스트 자동 문항 리스크 분석 (붙여넣기 전용)")
st.caption(
    "보안환경(업로드/GitHub 불가) 대응: 결과 테이블을 그대로 붙여넣으면 "
    "문항 리스크/원인 후보/KPI 대시보드를 자동 생성합니다."
)

# -----------------------------
# Utilities
# -----------------------------
def split_by_separator(raw: str):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    parts = re.split(r"\n\s*ㅡㅡ\s*\n", raw)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def detect_sep(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "\t"
    first = lines[0]
    if first.count("\t") >= max(2, first.count(",")):
        return "\t"
    return ","

def read_pasted_table(text: str) -> pd.DataFrame | None:
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    sep = detect_sep(t)
    try:
        df = pd.read_csv(StringIO(t), sep=sep)
        return df
    except Exception:
        for s in ["\t", ","]:
            try:
                df = pd.read_csv(StringIO(t), sep=s)
                return df
            except Exception:
                continue
        raise ValueError("붙여넣은 텍스트를 표(CSV/TSV)로 파싱할 수 없습니다.")

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def try_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def base_num(colname: str):
    s = str(colname).strip()
    m = re.match(r"^(10|[1-9])(\..+)?$", s)
    if not m:
        return None
    return int(m.group(1))

def suffix_key(colname: str):
    s = str(colname).strip()
    if "." in s:
        return s[s.find("."):]
    return ""

def pick_best_itemscore_group(df: pd.DataFrame, n_items=10):
    cols = list(df.columns)
    numbered = [c for c in cols if base_num(c) is not None]
    if not numbered:
        return None, []

    groups = {}
    for c in numbered:
        sk = suffix_key(c)
        groups.setdefault(sk, []).append(c)

    best = None
    best_score = -1

    for sk, gcols in groups.items():
        have = {base_num(c) for c in gcols}
        cover = sum(i in have for i in range(1, n_items + 1))
        sub = df[gcols].copy()
        numeric = sub.apply(try_num)
        numeric_rate = np.isfinite(numeric.values).mean()
        in_range = ((numeric >= 0) & (numeric <= 10)).to_numpy()
        range_rate = np.nanmean(in_range)
        score = cover * 10 + numeric_rate * 5 + range_rate * 5
        if score > best_score:
            best_score = score
            best = (sk, gcols)

    sk, gcols = best
    gcols_sorted = sorted(gcols, key=lambda c: base_num(c))
    return sk, gcols_sorted

def cronbach_alpha(item_df: pd.DataFrame) -> float:
    x = item_df.astype(float).dropna(axis=0, how="any")
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.nan
    k = x.shape[1]
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))

def point_biserial(item_score: pd.Series, total_excl: pd.Series) -> float:
    x = pd.to_numeric(item_score, errors="coerce")
    y = pd.to_numeric(total_excl, errors="coerce")
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 5:
        return np.nan
    if df["x"].std(ddof=0) == 0 or df["y"].std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(df["x"], df["y"])[0, 1])

def top_bottom_D(item_score: pd.Series, total_excl: pd.Series, frac=0.27) -> float:
    x = pd.to_numeric(item_score, errors="coerce")
    y = pd.to_numeric(total_excl, errors="coerce")
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 8:
        return np.nan
    df = df.sort_values("y")
    k = max(2, int(np.floor(n * frac)))
    low = df.head(k)["x"].mean()
    high = df.tail(k)["x"].mean()
    return float(high - low)

def risk_grade(p, pbis, D, cfg):
    reasons = []
    grade = "OK"

    if pd.notna(p) and p < cfg["p_hold"]:
        grade = "HOLD"
        reasons.append(f"정답률 매우 낮음(p<{cfg['p_hold']:.2f})")

    if pd.notna(p) and p < cfg["p_warn_low"]:
        if grade == "OK":
            grade = "WARN"
        reasons.append(f"정답률 낮음(p<{cfg['p_warn_low']:.2f})")

    if pd.notna(p) and p > cfg["p_too_easy"]:
        if grade == "OK":
            grade = "WARN"
        reasons.append(f"너무 쉬움(p>{cfg['p_too_easy']:.2f})")

    if pd.notna(pbis) and pbis < cfg["pb_warn"]:
        if grade == "OK":
            grade = "WARN"
        reasons.append(f"변별 낮음(pbis<{cfg['pb_warn']:.2f})")

    if pd.notna(D) and D < cfg["D_warn"]:
        if grade == "OK":
            grade = "WARN"
        reasons.append(f"상·하위 차이 낮음(D<{cfg['D_warn']:.2f})")

    if not reasons:
        reasons = ["-"]
    return grade, "; ".join(reasons)

def classify_cause(row, cfg):
    p = row["p_value"]
    pb = row["pbis"]
    D = row["D"]

    if pd.notna(pb) and pb < 0:
        return "🔴 문항결함_강의심(역변별: 아는사람이 더 틀림)"
    if pd.notna(D) and D < 0:
        return "🔴 문항결함_강의심(역상하위: 상위그룹이 더 낮음)"
    if pd.notna(pb) and pb < cfg["pb_warn"] and pd.notna(p) and p < cfg["p_warn_low"]:
        return "🟠 문항결함_약의심(변별없고 난이도도 높음 → 문항 모호 가능)"
    if pd.notna(p) and p < cfg["p_warn_low"] and pd.notna(pb) and pb >= cfg["pb_warn"]:
        return "🟡 역량부족_의심(어렵지만 변별됨 → 교육 강화 필요)"
    if pd.notna(p) and p > cfg["p_too_easy"]:
        return "🔵 천장효과(너무 쉬움 → 변별력 낮을 수 있음)"
    return "✅ 정상"

def stdize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    mapping = {
        "고유번호": "uid",
        "월": "month",
        "팀": "team",
        "직무": "role",
        "상담사": "name",
        "입사일": "hire_date",
        "근속개월": "tenure_group",
        "점수": "total_score",
    }
    for k, v in mapping.items():
        if k in df.columns:
            rename[k] = v
    df = df.rename(columns=rename)
    return df

def month_sort_key(m):
    if pd.isna(m):
        return "9999-99"
    s = str(m)
    s2 = re.sub(r"\s+", "", s)
    m1 = re.match(r"(\d{4})년(\d{1,2})월", s2)
    if m1:
        y = int(m1.group(1))
        mm = int(m1.group(2))
        return f"{y:04d}-{mm:02d}"
    return s

# -----------------------------
# Sidebar thresholds
# -----------------------------
st.sidebar.header("판정 기준(운영용)")
cfg = {
    "cut_score":   st.sidebar.number_input("기준점수(컷)", 0.0, 1000.0, 70.0, 0.5),
    "p_hold":      st.sidebar.slider("문항 HOLD 기준(정답률 p <)", 0.0, 0.6, 0.20, 0.01),
    "p_warn_low":  st.sidebar.slider("문항 경고 기준(정답률 p <)", 0.0, 0.8, 0.30, 0.01),
    "p_too_easy":  st.sidebar.slider("너무 쉬움 기준(정답률 p >)", 0.6, 1.0, 0.90, 0.01),
    "pb_warn":     st.sidebar.slider("변별 경고(pbis <)", -0.2, 0.6, 0.10, 0.01),
    "D_warn":      st.sidebar.slider("상·하위 차이 경고(D <)", -0.2, 1.0, 0.20, 0.01),
}

with st.expander("입력 안내(당신이 올릴 데이터 형태 기준)", expanded=False):
    st.markdown(
        """
- **박스1(메인)**: `고유번호/월/팀/직무/상담사/입사일/근속개월/점수` + `1~10` 반복 컬럼(문항점수/정오표/OX/텍스트 등) 포함 테이블을 그대로 붙여넣기
- **박스2(선택)**: `응답자ID...`로 시작하는 구글폼 원본 응답 테이블(선택, 분석 근거/예시 표시용)
- 앱은 **1~10이 반복되는 컬럼 중 "숫자(0~10/부분점수)가 가장 잘 파싱되는 그룹"을 자동으로 문항점수로 선택**합니다.
        """
    )

# -----------------------------
# Main inputs
# -----------------------------
st.subheader("1) 데이터 붙여넣기 (2개 박스)")
c1, c2 = st.columns(2)

with c1:
    main_text = st.text_area(
        "박스1) 결과/현황 테이블(월별 점수 + 1~10 반복 컬럼 포함) 붙여넣기",
        height=340,
        placeholder="여기에 탭 구분(TSV) 형태로 붙여넣으세요.",
    )
with c2:
    form_text = st.text_area(
        "박스2) (선택) 구글폼 원본 응답 테이블 붙여넣기",
        height=340,
        placeholder="응답자ID...로 시작하는 원본 표가 있으면 붙여넣기(선택).",
    )

run = st.button("분석 실행", type="primary")

# -----------------------------
# Analysis
# -----------------------------
if run:
    if not main_text.strip():
        st.error("박스1(메인 테이블)이 비어 있습니다.")
        st.stop()

    main_parts = split_by_separator(main_text)
    main_block = main_parts[0] if main_parts else main_text.strip()

    df_main = read_pasted_table(main_block)
    df_main = clean_cols(df_main)
    df_main = stdize_columns(df_main)

    suffix, score_cols = pick_best_itemscore_group(df_main, n_items=10)

    if not score_cols or len(score_cols) < 10:
        st.error("메인 테이블에서 1~10 문항 컬럼 그룹(점수)을 자동 식별하지 못했습니다.")
        st.stop()

    # -----------------------------
    # 2) 문항점수 그룹 선택
    # -----------------------------
    st.subheader("2) 문항점수(1~10) 그룹 자동 선택 결과")
    st.write(f"자동 선택된 그룹 suffix: `{suffix if suffix else '(no suffix)'}`, 컬럼: {score_cols}")

    numbered = [c for c in df_main.columns if base_num(c) is not None]
    groups = {}
    for c in numbered:
        groups.setdefault(suffix_key(c), []).append(c)
    group_keys = sorted(groups.keys(), key=lambda x: (len(x), x))
    group_label_map = {k: f"{k if k else '(no suffix)'} ({len(groups[k])} cols)" for k in group_keys}

    chosen_key = st.selectbox(
        "원하면 다른 1~10 그룹으로 변경",
        options=group_keys,
        format_func=lambda k: group_label_map[k],
        index=group_keys.index(suffix) if suffix in group_keys else 0
    )
    score_cols = sorted(groups[chosen_key], key=lambda c: base_num(c))

    item_score_df = df_main[score_cols].copy()
    item_score_df = item_score_df.apply(try_num)

    if "total_score" not in df_main.columns:
        df_main["total_score"] = item_score_df.sum(axis=1, min_count=1)

    df_main["total_score"] = pd.to_numeric(df_main["total_score"], errors="coerce")

    qcols = [f"Q{i}_score" for i in range(1, 11)]
    item_score_df.columns = qcols

    points = {}
    for q in qcols:
        mx = float(np.nanmax(item_score_df[q].values)) if np.isfinite(item_score_df[q].values).any() else 10.0
        points[q] = mx if mx > 0 else 10.0

    base_cols = [c for c in ["uid", "month", "team", "role", "name", "hire_date", "tenure_group", "total_score"] if c in df_main.columns]
    base = df_main[base_cols].copy()

    if "month" in base.columns:
        base["month_key"] = base["month"].map(month_sort_key)
    else:
        base["month"] = "NA"
        base["month_key"] = "NA"

    # -----------------------------
    # 3) KPI
    # -----------------------------
    st.subheader("3) KPI 대시보드 (시험 직후 자동 모니터링 6개)")

    n      = int(base["total_score"].notna().sum())
    mean_  = float(base["total_score"].mean())
    median_= float(base["total_score"].median())
    std_   = float(base["total_score"].std(ddof=1)) if n > 1 else np.nan
    p80    = float((base["total_score"] >= 80).mean())
    alpha  = cronbach_alpha(item_score_df.fillna(0))

    seg_gap = np.nan
    if "team" in base.columns and "role" in base.columns:
        seg = base.dropna(subset=["team", "role", "total_score"]).copy()
        if len(seg) >= 2:
            seg["seg"] = seg["team"].astype(str) + " / " + seg["role"].astype(str)
            means = seg.groupby("seg")["total_score"].mean()
            if len(means) >= 2:
                seg_gap = float(means.max() - means.min())

    # item stats
    item_rows = []
    total = base["total_score"].copy()

    for i, q in enumerate(qcols, start=1):
        pts   = points[q]
        sc    = item_score_df[q]
        ratio = sc / pts if pts > 0 else np.nan
        p_value = float(ratio.mean())
        total_excl = total - sc.fillna(0)
        pb    = point_biserial(sc, total_excl)
        D     = top_bottom_D(sc, total_excl)
        grade, reason = risk_grade(p_value, pb, D, cfg)
        item_rows.append({
            "item": i,
            "q": q.replace("_score", ""),
            "points_est": round(pts, 2),
            "p_value": p_value,
            "pbis": pb,
            "D": D,
            "risk_grade": grade,
            "reasons": reason
        })

    item_stats = pd.DataFrame(item_rows)
    item_fail_rate = float(item_stats["risk_grade"].isin(["HOLD"]).mean())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("평균",          f"{mean_:.2f}",                           help="전체 평균(총점)")
    k2.metric("중앙값",        f"{median_:.2f}",                         help="중앙값(총점)")
    k3.metric("≥80 비율",      f"{p80*100:.1f}%",                        help="상단 유지(상위권 붕괴 감지)")
    k4.metric("신뢰도(α)",     f"{alpha:.2f}" if pd.notna(alpha) else "-",help="Cronbach alpha(아이템점수 기반)")
    k5.metric("문항 HOLD 비율",f"{item_fail_rate*100:.1f}%",             help="HOLD(반영보류 권고) 문항 비율")
    k6.metric("세그 격차",     f"{seg_gap:.1f}" if pd.notna(seg_gap) else "-", help="팀×직무 평균 max-min")

    cA, cB = st.columns([1.2, 1.0])
    with cA:
        fig = px.histogram(base.dropna(subset=["total_score"]), x="total_score", nbins=12, title="총점 분포")
        fig.update_layout(height=320, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        grade_counts = (
            item_stats["risk_grade"]
            .value_counts()
            .reindex(["HOLD", "WARN", "OK"])
            .fillna(0).astype(int)
            .reset_index()
        )
        grade_counts.columns = ["risk_grade", "count"]
        fig2 = px.bar(grade_counts, x="risk_grade", y="count", title="문항 리스크 등급 분포")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # 4) 월별 트렌드
    # -----------------------------
    st.subheader("4) 월별 트렌드(가능한 경우)")
    if "month" in base.columns and base["month"].nunique() >= 2:
        trend = (
            base.groupby(["month", "month_key"])["total_score"]
            .agg(n="count", mean="mean", median="median", std="std")
            .reset_index()
            .sort_values("month_key")
        )
        trend["p80"]       = base.groupby(["month", "month_key"]).apply(lambda g: (g["total_score"] >= 80).mean()).values
        trend["pass_rate"] = base.groupby(["month", "month_key"]).apply(lambda g: (g["total_score"] >= cfg["cut_score"]).mean()).values

        t1, t2 = st.columns(2)
        with t1:
            figt = px.line(trend, x="month", y="mean", markers=True, title="월별 평균 추이")
            figt.update_layout(height=300)
            st.plotly_chart(figt, use_container_width=True)
        with t2:
            figt2 = px.line(trend, x="month", y="p80", markers=True, title="월별 ≥80 비율 추이")
            figt2.update_layout(height=300, yaxis_tickformat=".0%")
            st.plotly_chart(figt2, use_container_width=True)
        st.dataframe(trend.round(3), use_container_width=True)
    else:
        st.info("월 컬럼이 없거나 단일 월만 존재합니다. (월별 추세는 생략)")

    # -----------------------------
    # 5) 문항 리스크 테이블
    # -----------------------------
    st.subheader("5) 문항 리스크 분석(난이도/변별/상하위차) + HOLD 후보")
    show = item_stats.copy()
    show["p_value"] = show["p_value"].round(3)
    show["pbis"]    = show["pbis"].round(3)
    show["D"]       = show["D"].round(3)
    st.dataframe(
        show.sort_values(["risk_grade", "p_value"], ascending=[True, True]),
        use_container_width=True
    )

    hold_items = item_stats[item_stats["risk_grade"] == "HOLD"]
    if len(hold_items):
        st.warning(
            "HOLD(반영보류 권고) 문항이 있습니다. "
            "이 문항들은 평균 급락/공정성 논란의 1차 후보이므로 즉시 검토(채점/문항 모호/부분점수) 권장."
        )

    # -----------------------------------------------
    # 5-1) 원인 분리: 문항 결함 vs 사람 문제
    # -----------------------------------------------
    st.subheader("5-1) 원인 분리 분석: 문항 결함 vs 응시자 역량 부족")
    st.markdown(
        """
| 아이콘 | 유형 | 핵심 신호 | 권장 조치 |
|--------|------|-----------|-----------|
| 🔴 | 문항결함_강의심 | pbis < 0 또는 D < 0 (잘 아는 사람도 틀림) | 정답키·오탈자·복수정답 즉시 확인 |
| 🟠 | 문항결함_약의심 | pbis 낮음 + p 낮음 (모호한 문항) | 문항 표현·범위 재검토 |
| 🟡 | 역량부족_의심 | p 낮지만 pbis 정상 (어렵지만 구분됨) | 해당 영역 교육 강화 |
| 🔵 | 천장효과 | p 너무 높음 (너무 쉬움) | 난이도 상향 검토 |
| ✅ | 정상 | 모두 기준 통과 | 유지 |
        """
    )

    item_stats["cause_type"] = item_stats.apply(lambda row: classify_cause(row, cfg), axis=1)

    cause_show = item_stats[["item", "q", "p_value", "pbis", "D", "risk_grade", "cause_type", "reasons"]].copy()
    cause_show["p_value"] = cause_show["p_value"].round(3)
    cause_show["pbis"]    = cause_show["pbis"].round(3)
    cause_show["D"]       = cause_show["D"].round(3)
    st.dataframe(
        cause_show.sort_values(["risk_grade", "p_value"], ascending=[True, True]),
        use_container_width=True
    )

    cause_counts = item_stats["cause_type"].value_counts().reset_index()
    cause_counts.columns = ["원인유형", "문항수"]

    color_map = {
        "🔴 문항결함_강의심(역변별: 아는사람이 더 틀림)":         "#d62728",
        "🔴 문항결함_강의심(역상하위: 상위그룹이 더 낮음)":        "#d62728",
        "🟠 문항결함_약의심(변별없고 난이도도 높음 → 문항 모호 가능)": "#ff7f0e",
        "🟡 역량부족_의심(어렵지만 변별됨 → 교육 강화 필요)":      "#f5c518",
        "🔵 천장효과(너무 쉬움 → 변별력 낮을 수 있음)":            "#1f77b4",
        "✅ 정상":                                                  "#2ca02c",
    }
    fig_cause = px.bar(
        cause_counts, x="문항수", y="원인유형", orientation="h",
        title="문항별 원인 유형 분포",
        color="원인유형",
        color_discrete_map=color_map,
    )
    fig_cause.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_cause, use_container_width=True)

    # -----------------------------------------------
    # 5-2) 세그×문항 히트맵
    # -----------------------------------------------
    st.subheader("5-2) 세그(팀/직무)별 문항 점수 히트맵: 누가 어떤 문항에서 약한가")

    seg_col_for_heatmap = None
    for candidate in ["team", "role", "tenure_group"]:
        if candidate in df_main.columns:
            seg_col_for_heatmap = candidate
            break

    if seg_col_for_heatmap:
        merged = pd.concat(
            [df_main[[seg_col_for_heatmap]].reset_index(drop=True),
             item_score_df.reset_index(drop=True)],
            axis=1
        )
        merged = merged.dropna(subset=[seg_col_for_heatmap])

        ratio_df = merged.copy()
        for q in qcols:
            pts = points[q]
            ratio_df[q] = merged[q] / pts if pts > 0 else np.nan

        heatmap_data = ratio_df.groupby(seg_col_for_heatmap)[qcols].mean()
        heatmap_data.columns = [c.replace("_score", "") for c in heatmap_data.columns]

        fig_hm = px.imshow(
            heatmap_data,
            text_auto=".0%",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title=f"{seg_col_for_heatmap}별 × 문항별 정답률 히트맵 (🔴빨강=취약 / 🟢초록=강함)"
        )
        fig_hm.update_layout(height=420)
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(
            "✅ 특정 문항 열 전체가 빨간색 → 문항 결함 의심  |  "
            "특정 행(그룹) 전체가 빨간색 → 해당 그룹 역량 부족 의심"
        )

        # 세그 선택 드롭다운 추가
        seg_all_options = [c for c in ["team", "role", "tenure_group"] if c in df_main.columns]
        if len(seg_all_options) > 1:
            seg_col_for_heatmap = st.selectbox(
                "히트맵 세그 기준 변경",
                options=seg_all_options,
                index=0
            )
            merged2 = pd.concat(
                [df_main[[seg_col_for_heatmap]].reset_index(drop=True),
                 item_score_df.reset_index(drop=True)],
                axis=1
            )
            merged2 = merged2.dropna(subset=[seg_col_for_heatmap])
            ratio_df2 = merged2.copy()
            for q in qcols:
                pts = points[q]
                ratio_df2[q] = merged2[q] / pts if pts > 0 else np.nan
            heatmap_data2 = ratio_df2.groupby(seg_col_for_heatmap)[qcols].mean()
            heatmap_data2.columns = [c.replace("_score", "") for c in heatmap_data2.columns]
            fig_hm2 = px.imshow(
                heatmap_data2,
                text_auto=".0%",
                aspect="auto",
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=1,
                title=f"{seg_col_for_heatmap}별 × 문항별 정답률 히트맵"
            )
            fig_hm2.update_layout(height=420)
            st.plotly_chart(fig_hm2, use_container_width=True)
    else:
        st.info("팀/직무 컬럼이 있으면 세그별 히트맵이 표시됩니다.")

    # -----------------------------------------------
    # 5-3) 자동 종합 인사이트
    # -----------------------------------------------
    st.subheader("5-3) 자동 종합 인사이트")

    insights = []

    defect_strong = item_stats[item_stats["cause_type"].str.startswith("🔴")]
    if len(defect_strong):
        qs = ", ".join(defect_strong["q"].tolist())
        insights.append(
            f"🔴 **문항 결함 강의심 ({qs})**: 총점이 높은(잘 아는) 응시자도 틀리는 패턴입니다. "
            f"정답키 오류·오탈자·복수정답을 먼저 확인하세요."
        )

    defect_weak = item_stats[item_stats["cause_type"].str.startswith("🟠")]
    if len(defect_weak):
        qs = ", ".join(defect_weak["q"].tolist())
        insights.append(
            f"🟠 **문항 모호성 의심 ({qs})**: 정답률도 낮고 변별도 없습니다. "
            f"문항이 모호하거나 교육에서 다루지 않은 내용일 수 있습니다."
        )

    skill_lack = item_stats[item_stats["cause_type"].str.startswith("🟡")]
    if len(skill_lack):
        qs = ", ".join(skill_lack["q"].tolist())
        insights.append(
            f"🟡 **응시자 역량 부족 의심 ({qs})**: 어렵지만 아는 사람/모르는 사람 구분은 됩니다. "
            f"이 영역에 대한 교육 강화를 검토하세요."
        )

    ceiling = item_stats[item_stats["cause_type"].str.startswith("🔵")]
    if len(ceiling):
        qs = ", ".join(ceiling["q"].tolist())
        insights.append(
            f"🔵 **천장효과 ({qs})**: 너무 쉬워 변별력이 낮습니다. "
            f"난이도 상향 또는 문항 교체를 검토하세요."
        )

    if mean_ < cfg["cut_score"]:
        insights.append(
            f"📉 **전체 평균({mean_:.1f})이 기준점({cfg['cut_score']:.0f}) 미만**입니다. "
            f"문항 결함 문항 제외 후 재산정하거나, 전반적 교육 수준을 점검하세요."
        )

    if pd.notna(alpha) and alpha < 0.6:
        insights.append(
            f"⚠️ **신뢰도(α={alpha:.2f})가 낮습니다.** 문항들이 동일한 개념을 측정하지 못하고 있을 수 있습니다. "
            f"결함 문항 제거 후 α를 재산정하세요."
        )

    if not insights:
        insights.append("✅ 현재 기준상 심각한 문항 결함 또는 역량 이슈가 감지되지 않았습니다.")

    for insight in insights:
        st.markdown(f"- {insight}")

    # -----------------------------
    # 6) 시뮬레이션
    # -----------------------------
    st.subheader("6) 원인분해(시험문항 영향도) 시뮬레이션: HOLD 문항 제외 시 평균 회복")
    hold_qs = set(hold_items["q"].tolist())
    if hold_qs:
        cols_map = {f"Q{i}": f"Q{i}_score" for i in range(1, 11)}
        hold_score_cols = [cols_map[q] for q in hold_qs if q in cols_map]
        adj_total = base["total_score"] - item_score_df[hold_score_cols].sum(axis=1, min_count=1)
        st.write(
            f"- HOLD 문항: {sorted(list(hold_qs))}\n"
            f"- 현재 평균: {mean_:.2f}\n"
            f"- HOLD 문항 점수 제외 평균(참고): {float(adj_total.mean()):.2f}\n"
            f"- 평균 차이(참고): {float(adj_total.mean() - mean_):+.2f}"
        )
        st.caption("참고: 위는 '점수 제외' 기준의 빠른 영향도입니다. 실제 운영은 '문항 무효/재채점/부분점수' 정책에 따라 재산정하세요.")
    else:
        st.info("HOLD 문항이 없어 시뮬레이션을 생략합니다.")

    # -----------------------------
    # 7) 세그 구조 패턴
    # -----------------------------
    st.subheader("7) 세그(팀/직무/근속) 구조 패턴")
    seg_tabs = [col for col in ["team", "role", "tenure_group"] if col in base.columns]

    if seg_tabs:
        seg_choice = st.selectbox("세그 기준 선택", seg_tabs, index=0)
        seg = (
            base.dropna(subset=[seg_choice, "total_score"])
            .groupby(seg_choice)["total_score"]
            .agg(n="count", mean="mean", median="median", std="std")
            .reset_index()
            .sort_values("mean")
        )
        seg["p80"]       = base.dropna(subset=[seg_choice]).groupby(seg_choice).apply(lambda g: (g["total_score"] >= 80).mean()).values
        seg["pass_rate"] = base.dropna(subset=[seg_choice]).groupby(seg_choice).apply(lambda g: (g["total_score"] >= cfg["cut_score"]).mean()).values

        s1, s2 = st.columns([1.2, 1.0])
        with s1:
            st.dataframe(seg.round(3), use_container_width=True)
        with s2:
            figS = px.bar(seg, x="mean", y=seg_choice, orientation="h",
                          title=f"{seg_choice}별 평균(총점)", text="n")
            figS.update_layout(height=360)
            st.plotly_chart(figS, use_container_width=True)

        if "team" in base.columns and "role" in base.columns:
            pivot = base.pivot_table(index="team", columns="role", values="total_score", aggfunc="mean")
            figH = px.imshow(pivot, text_auto=True, aspect="auto", title="팀×직무 평균(총점) 히트맵")
            figH.update_layout(height=380)
            st.plotly_chart(figH, use_container_width=True)
    else:
        st.info("팀/직무/근속 컬럼이 없어 세그 분석을 생략합니다.")

    # -----------------------------
    # 8) 응답자별 인사이트
    # -----------------------------
    st.subheader("8) 응답자별 인사이트(취약 문항 TOP3)")
    indiv = base.copy()
    gaps = []
    for i, q in enumerate(qcols, start=1):
        pts   = points[q]
        ratio = (item_score_df[q] / pts) if pts > 0 else np.nan
        gaps.append(1 - ratio)
    gap_df = pd.concat(gaps, axis=1)
    gap_df.columns = [f"Q{i}_gap" for i in range(1, 11)]

    def top3_str(row):
        pairs = []
        for i in range(1, 11):
            v = row.get(f"Q{i}_gap")
            if pd.isna(v):
                continue
            pairs.append((i, float(v)))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:3]
        if not pairs:
            return "-"
        return "; ".join([f"Q{i}({g:.0%})" for i, g in pairs])

    indiv["weak_top3"] = gap_df.apply(top3_str, axis=1)
    show_cols = [c for c in ["month", "team", "role", "tenure_group", "name", "uid", "total_score", "weak_top3"] if c in indiv.columns]
    st.dataframe(indiv[show_cols].sort_values("total_score"), use_container_width=True)

    # -----------------------------
    # 9) 구글폼 원본
    # -----------------------------
    st.subheader("9) (선택) 구글폼 원본 응답 테이블 확인")
    if form_text.strip():
        form_parts = split_by_separator(form_text)
        form_block = form_parts[0] if form_parts else form_text.strip()
        try:
            df_form = read_pasted_table(form_block)
            df_form = clean_cols(df_form)
            st.success("구글폼 원본 표 파싱 성공(참고용 표시)")
            st.dataframe(df_form.head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"구글폼 원본 표 파싱 실패(분석에는 영향 없음): {e}")
    else:
        st.info("박스2는 선택입니다. 비워도 분석은 가능합니다.")

    # -----------------------------
    # 10) 다운로드
    # -----------------------------
    st.subheader("10) 결과 다운로드(CSV)")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "문항 통계(item_stats) CSV",
            data=item_stats.to_csv(index=False).encode("utf-8-sig"),
            file_name="item_stats.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "개인 인사이트(individuals) CSV",
            data=indiv[show_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="individual_insights.csv",
            mime="text/csv",
        )
    with d3:
        long = pd.DataFrame({"idx": np.arange(len(base))})
        for q in qcols:
            long[q.replace("_score", "")] = item_score_df[q]
        long = pd.concat([base.reset_index(drop=True), long.drop(columns=["idx"])], axis=1)
        st.download_button(
            "응답자×문항점수(wide) CSV",
            data=long.to_csv(index=False).encode("utf-8-sig"),
            file_name="scoring_wide.csv",
            mime="text/csv",
        )

else:
    st.info("박스1에 데이터 붙여넣고 [분석 실행]을 누르세요.")

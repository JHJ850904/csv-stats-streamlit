# streamlit_app.py
# ============================================
# 실행:  python -m streamlit run streamlit_app.py
# 필요 패키지:
#   pip install streamlit pandas numpy scipy scikit-learn statsmodels plotly python-dateutil
# ============================================

from __future__ import annotations
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dateutil.parser import parse as dt_parse
from scipy import stats

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    r2_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="CSV 통계분석 웹앱 (베타)", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
#  유틸: 타입 추론
# =============================================================
@dataclass
class InferredTypes:
    numeric: List[str]
    categorical: List[str]
    boolean: List[str]
    datetime: List[str]


def _looks_datetime(series: pd.Series, sample: int = 50) -> bool:
    if series.dtype.kind in ("M",):
        return True
    if series.dtype == "object":
        ex = series.dropna().astype(str).head(sample)
        ok = 0
        for v in ex:
            try:
                _ = dt_parse(v, fuzzy=False)
                ok += 1
            except Exception:
                pass
        return ok >= max(3, len(ex) // 2)
    return False


def infer_types(df: pd.DataFrame) -> InferredTypes:
    numeric: List[str] = []
    categorical: List[str] = []
    boolean: List[str] = []
    datetime_cols: List[str] = []

    for c in df.columns:
        s = df[c]
        if s.dropna().isin([0, 1, True, False, "0", "1", "true", "false"]).all():
            boolean.append(c)
            continue
        if _looks_datetime(s):
            datetime_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            if nunique <= max(10, int(0.02 * len(s))):
                categorical.append(c)
            else:
                numeric.append(c)
        else:
            categorical.append(c)
    return InferredTypes(numeric=numeric, categorical=categorical, boolean=boolean, datetime=datetime_cols)

# =============================================================
#  효과크기 & 해석 유틸
# =============================================================

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s if s != 0 else 0.0


def eta_squared_anova(groups: List[np.ndarray]) -> float:
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    sst = ((all_values - grand_mean) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return float(np.sqrt(chi2 / (n * (min(r - 1, c - 1))))) if min(r, c) > 1 else np.nan


def _sig_text(p: float) -> str:
    return "유의함 (p<0.05)" if (p is not None and p < 0.05) else "유의하지 않음 (p≥0.05)"


def _cohen_d_level(d: float) -> str:
    if d is None or not np.isfinite(d):
        return "–"
    ad = abs(d)
    return "매우 작음" if ad < 0.2 else ("작음" if ad < 0.5 else ("중간" if ad < 0.8 else "큼"))


def _eta2_level(e: float) -> str:
    if e is None or not np.isfinite(e):
        return "–"
    return "매우 작음" if e < 0.01 else ("작음" if e < 0.06 else ("중간" if e < 0.14 else "큼"))


def _cramers_v_level(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    av = abs(v)
    return "약함" if av < 0.1 else ("보통" if av < 0.3 else ("강함" if av < 0.5 else "매우 강함"))


def _r2_level(r2: float) -> str:
    if r2 is None or not np.isfinite(r2):
        return "–"
    return "약함" if r2 < 0.3 else ("보통" if r2 < 0.5 else ("강함" if r2 < 0.7 else "매우 강함"))


def _auc_level(auc: float) -> str:
    if auc is None or not np.isfinite(auc):
        return "–"
    return "무작위에 가까움" if auc < 0.6 else ("낮음" if auc < 0.7 else ("보통" if auc < 0.8 else ("좋음" if auc < 0.9 else "매우 좋음")))

# =============================================================
#  시각적 해석 도구
# =============================================================

def create_significance_badge(p_value: float) -> str:
    """통계적 유의성 배지 생성"""
    if p_value < 0.001:
        return "🟩 **매우 유의함** (p<0.001)"
    elif p_value < 0.01:
        return "🟢 **유의함** (p<0.01)"
    elif p_value < 0.05:
        return "🟡 **유의함** (p<0.05)"
    elif p_value < 0.1:
        return "🟠 **경계적** (p<0.1)"
    else:
        return "🔴 **유의하지 않음** (p≥0.1)"

def create_effect_size_badge(effect_size: float, effect_type: str) -> str:
    """효과크기 배지 생성"""
    if effect_type == "cohen_d":
        abs_effect = abs(effect_size) if np.isfinite(effect_size) else 0
        if abs_effect >= 0.8:
            return "🔥 **큰 효과**"
        elif abs_effect >= 0.5:
            return "📈 **중간 효과**"
        elif abs_effect >= 0.2:
            return "📊 **작은 효과**"
        else:
            return "📉 **무시할 수 있는 효과**"
    elif effect_type == "eta_squared":
        if effect_size >= 0.14:
            return "🔥 **큰 효과**"
        elif effect_size >= 0.06:
            return "📈 **중간 효과**"
        elif effect_size >= 0.01:
            return "📊 **작은 효과**"
        else:
            return "📉 **무시할 수 있는 효과**"
    elif effect_type == "r_squared":
        if effect_size >= 0.7:
            return "🔥 **매우 강한 설명력**"
        elif effect_size >= 0.5:
            return "📈 **강한 설명력**"
        elif effect_size >= 0.3:
            return "📊 **보통 설명력**"
        else:
            return "📉 **약한 설명력**"
    return ""

def create_correlation_badge(corr_value: float) -> str:
    """상관계수 배지 생성"""
    abs_corr = abs(corr_value) if np.isfinite(corr_value) else 0
    if abs_corr >= 0.8:
        return "🔥 **매우 강한 상관**"
    elif abs_corr >= 0.6:
        return "📈 **강한 상관**"
    elif abs_corr >= 0.4:
        return "📊 **중간 상관**"
    elif abs_corr >= 0.2:
        return "📉 **약한 상관**"
    else:
        return "➖ **거의 무상관**"

def create_progress_bar(value: float, max_value: float, label: str) -> str:
    """진행률 바 생성"""
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    percentage = min(100, max(0, percentage))
    
    filled_blocks = int(percentage / 10)
    empty_blocks = 10 - filled_blocks
    
    bar = "█" * filled_blocks + "░" * empty_blocks
    return f"{label}: {bar} {percentage:.1f}%"

def display_metric_card(title: str, value: str, delta: str = None, color: str = "normal"):
    """메트릭 카드 표시"""
    if color == "good":
        st.success(f"**{title}**: {value}" + (f" ({delta})" if delta else ""))
    elif color == "warning":
        st.warning(f"**{title}**: {value}" + (f" ({delta})" if delta else ""))
    elif color == "error":
        st.error(f"**{title}**: {value}" + (f" ({delta})" if delta else ""))
    else:
        st.info(f"**{title}**: {value}" + (f" ({delta})" if delta else ""))

# =============================================================
#  로딩/전처리
# =============================================================
@st.cache_data(show_spinner=False)
def read_csv_safely(file, sep: Optional[str], encoding: Optional[str]) -> pd.DataFrame:
    kwargs = {}
    if sep:
        kwargs["sep"] = sep
    else:
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    if encoding:
        kwargs["encoding"] = encoding
    try:
        df = pd.read_csv(file, **kwargs)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, **{**kwargs, "encoding": "cp949"})
    return df

# 시각화 유틸

def show_distribution(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if pd.api.types.is_numeric_dtype(s):
        fig = px.histogram(s, x=col, nbins=30, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        vc = s.value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
#  분석 제안 로직
# =============================================================
@dataclass
class Suggestion:
    key: str
    label: str
    desc: str


def suggest_analyses(df: pd.DataFrame, it: InferredTypes, target: Optional[str]) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    suggestions.append(Suggestion("eda", "기초 EDA 요약", "컬럼 요약, 결측, 분포, 간단 시각화"))
    if len(it.numeric) >= 2:
        suggestions.append(Suggestion("correlation", "상관분석 (피어슨/스피어만)", "연속형 변수 간 상관행렬과 히트맵"))
    has_group_compare = any((df[c].nunique(dropna=True) >= 2 and df[c].nunique(dropna=True) <= 10) for c in (it.categorical + it.boolean)) and (len(it.numeric) >= 1)
    if has_group_compare:
        suggestions.append(Suggestion("group_compare", "그룹 간 평균 비교 (t-검정/ANOVA)", "범주형 × 연속형 조합으로 집단 비교"))
        suggestions.append(Suggestion("tukey", "사후검정 (Tukey HSD)", "유의하면 어떤 그룹끼리 차이인지 확인"))
    if len(it.categorical + it.boolean) >= 2:
        suggestions.append(Suggestion("chi_square", "카이제곱 독립성 검정", "두 범주형 변수의 독립성 검정"))
    if target is not None and target in df.columns:
        y = df[target]
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 2:
            suggestions.append(Suggestion("linreg", f"✅ 선형 회귀 (타겟: {target})", f"연속형 타겟 '{target}' 예측 모델"))
            suggestions.append(Suggestion("regdiag", "회귀 진단 (VIF/잔차)", "다중공선성·잔차 진단"))
            suggestions.append(Suggestion("featimp", f"✅ 특성 중요도 (타겟: {target})", f"'{target}'에 영향을 미치는 변수 분석"))
        else:
            suggestions.append(Suggestion("logreg", f"✅ 로지스틱 회귀 (타겟: {target})", f"범주형 타겟 '{target}' 분류 모델"))
            suggestions.append(Suggestion("featimp", f"✅ 특성 중요도 (타겟: {target})", f"'{target}'에 영향을 미치는 변수 분석"))
    else:
        # 타겟이 설정되지 않은 경우
        if len(it.numeric) >= 1:
            suggestions.append(Suggestion("linreg", "⚠️ 선형 회귀 (타겟 미설정)", "실행 시 연속형 타겟을 선택해야 함"))
        binary_cats = [c for c in (it.categorical + it.boolean) if df[c].nunique(dropna=True) == 2]
        if binary_cats:
            suggestions.append(Suggestion("logreg", "⚠️ 로지스틱 회귀 (타겟 미설정)", "실행 시 범주형 타겟을 선택해야 함"))
        suggestions.append(Suggestion("featimp", "⚠️ 특성 중요도 (타겟 미설정)", "실행 시 타겟을 선택해야 함"))
    if len(it.numeric) >= 2:
        suggestions.append(Suggestion("kmeans", "KMeans 군집화", "연속형 변수들로 군집 탐색 (PCA 2D 시각화)"))
    if it.datetime and it.numeric:
        suggestions.append(Suggestion("timeseries", "시계열 추세/계절성", "날짜 기준 집계 후 추세·계절성 확인"))
    suggestions.append(Suggestion("insights", "자동 인사이트", "상관·결측·그룹차이 등 핵심 요약"))
    suggestions.append(Suggestion("outliers", "이상치 탐지", "IQR/z-score·Isolation Forest"))
    return suggestions

# =============================================================
#  전처리 파이프라인 빌드 (결측치 안전)
# =============================================================

def _build_preprocess(df: pd.DataFrame, x_cols: List[str]) -> Tuple[Pipeline, List[str]]:
    X = df[x_cols]
    num_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("pre", pre)])

    X_sample = X.head(2000)
    pre.fit(X_sample)
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = []
    return pipe, feat_names

# =============================================================
#  분석 함수들
# =============================================================

def run_correlation(df: pd.DataFrame, numeric_cols: List[str]):
    st.subheader("📈 상관분석")
    st.caption("피어슨: 선형관계 r, 스피어만: 순위 기반. |r|가 클수록 강함.")
    cols = st.multiselect("분석할 연속형 변수 선택", options=numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
    if len(cols) < 2:
        st.info("두 개 이상 선택하세요.")
        return "", None
    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")

    # 상관분석 요약 대시보드
    st.markdown("### 📊 상관분석 요약")
    
    try:
        a = corr_p.abs()
        upper = a.where(np.triu(np.ones(a.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        
        if len(pairs) > 0:
            max_corr = pairs.max()
            strong_pairs = (pairs >= 0.8).sum()
            moderate_pairs = ((pairs >= 0.5) & (pairs < 0.8)).sum()
            weak_pairs = (pairs < 0.3).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                color = "good" if max_corr < 0.8 else "warning" if max_corr < 0.9 else "error"
                display_metric_card("최대 상관계수", f"{max_corr:.3f}", color=color)
            
            with col2:
                color = "error" if strong_pairs > 0 else "good"
                display_metric_card("강한 상관 (≥0.8)", f"{strong_pairs}개", color=color)
            
            with col3:
                display_metric_card("중간 상관 (0.5-0.8)", f"{moderate_pairs}개", color="normal")
            
            with col4:
                display_metric_card("약한 상관 (<0.3)", f"{weak_pairs}개", color="normal")

    except Exception:
        pass

    st.markdown("**피어슨 상관행렬**")
    fig1 = px.imshow(corr_p, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**스피어만 상관행렬**")
    fig2 = px.imshow(corr_s, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig2, use_container_width=True)

    # 상위 상관쌍 분석
    try:
        a = corr_p.abs()
        upper = a.where(np.triu(np.ones(a.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        top = pairs.head(min(5, len(pairs)))
        
        if len(top) > 0:
            st.markdown("### 🔍 주요 상관관계")
            for (i, j), v in top.items():
                corr_val = corr_p.loc[i, j]
                badge = create_correlation_badge(corr_val)
                direction = "양의 상관" if corr_val > 0 else "음의 상관"
                st.markdown(f"**{i} ↔ {j}**: {corr_val:.3f} {badge} ({direction})")
        
        strong_flag = (pairs.max() if len(pairs) > 0 else 0) >= 0.8
        if strong_flag:
            st.warning("⚠️ **주의**: |r|≥0.8인 강한 상관쌍이 존재합니다. 회귀분석 시 다중공선성을 확인하세요!")
        
        top_lines = [f"- {i}–{j}: r={corr_p.loc[i, j]:.2f} ({create_correlation_badge(corr_p.loc[i, j])})" for (i, j), v in top.items()]
    except Exception:
        top_lines, strong_flag = [], False

    md_lines = [
        "### 상관분석 결과",
        "피어슨/스피어만 상관을 계산했습니다.",
        "**주요 상관쌍:**",
    ] + (top_lines if top_lines else ["- (충분한 쌍이 없음)"])
    if strong_flag:
        md_lines.append("⚠️ |r|≥0.8인 강한 상관쌍 존재 → 다중공선성 주의")

    return "\n".join(md_lines), {"pearson": corr_p, "spearman": corr_s}


def run_group_compare(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]):
    st.subheader("🧪 그룹 간 평균 비교")
    st.caption("두 집단이면 t-검정/Mann–Whitney, 세 집단 이상이면 ANOVA/Kruskal-Wallis. p<0.05면 평균 차이가 유의하며, 효과크기 d/η²로 크기를 함께 봅니다.")
    num = st.selectbox("연속형 변수", options=numeric_cols)
    cat = st.selectbox("그룹(범주형) 변수", options=cat_cols)

    groups = df[[num, cat]].dropna()
    k = groups[cat].nunique()

    auto_nonparam = st.checkbox("정규성 위반 시 비모수 대체 (Shapiro / Mann-Whitney, Kruskal)", value=True)

    result_md: List[str] = []

    if k == 2:
        levels = groups[cat].dropna().unique()
        g1 = groups[groups[cat] == levels[0]][num].values
        g2 = groups[groups[cat] == levels[1]][num].values
        use_nonparam = False
        if auto_nonparam:
            for g in [g1, g2]:
                sample = g if len(g) <= 5000 else np.random.choice(g, 5000, replace=False)
                try:
                    sh_p = stats.shapiro(sample).pvalue
                    if sh_p < 0.05:
                        use_nonparam = True
                        break
                except Exception:
                    pass
        if use_nonparam:
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U"
            eff = np.nan
        else:
            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
            test_name = "독립표본 t-검정 (Welch)"
            eff = cohen_d(g1, g2)

        # 결과 대시보드
        st.markdown("### 📊 검정 결과 요약")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sig_badge = create_significance_badge(p)
            st.markdown(f"**통계적 유의성**\n\n{sig_badge}")
        
        with col2:
            if not np.isnan(eff):
                effect_badge = create_effect_size_badge(eff, "cohen_d")
                st.markdown(f"**효과 크기**\n\n{effect_badge}\n\nCohen's d = {eff:.3f}")
        
        with col3:
            mean1, mean2 = np.mean(g1), np.mean(g2)
            diff = abs(mean1 - mean2)
            st.markdown(f"**평균 차이**\n\n{diff:.3f}\n\n{levels[0]}: {mean1:.2f}\n{levels[1]}: {mean2:.2f}")

        result_md.append(f"**검정:** {test_name}\n\n**p-value:** {p:.4g}")
        if not np.isnan(eff):
            result_md.append(f"**효과크기 (Cohen's d):** {eff:.3f}")
        interp = f"**해석:** {_sig_text(p)}"
        if not np.isnan(eff):
            interp += f" / 효과크기: {eff:.2f} ({_cohen_d_level(eff)})"
        result_md.append(interp)

        fig = px.box(groups, x=cat, y=num, points="all", color=cat)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        data_groups = [groups[groups[cat] == lv][num].dropna().values for lv in groups[cat].unique()]
        use_nonparam = False
        if auto_nonparam:
            for g in data_groups:
                sample = g if len(g) <= 5000 else np.random.choice(g, 5000, replace=False)
                try:
                    sh_p = stats.shapiro(sample).pvalue
                    if sh_p < 0.05:
                        use_nonparam = True
                        break
                except Exception:
                    pass
        if use_nonparam:
            stat, p = stats.kruskal(*data_groups)
            test_name = "Kruskal-Wallis"
            eta2 = np.nan
        else:
            stat, p = stats.f_oneway(*data_groups)
            test_name = "일원분산분석 (ANOVA)"
            eta2 = eta_squared_anova(data_groups)

        # 결과 대시보드
        st.markdown("### 📊 검정 결과 요약")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sig_badge = create_significance_badge(p)
            st.markdown(f"**통계적 유의성**\n\n{sig_badge}")
        
        with col2:
            if not np.isnan(eta2):
                effect_badge = create_effect_size_badge(eta2, "eta_squared")
                st.markdown(f"**효과 크기**\n\n{effect_badge}\n\nEta² = {eta2:.3f}")
        
        with col3:
            group_means = [np.mean(g) for g in data_groups]
            overall_mean = np.mean([val for group in data_groups for val in group])
            variability = np.std(group_means)
            st.markdown(f"**그룹 간 변동성**\n\n{variability:.3f}\n\n전체 평균: {overall_mean:.2f}")

        result_md.append(f"**검정:** {test_name}\n\n**p-value:** {p:.4g}")
        if not np.isnan(eta2):
            result_md.append(f"**효과크기 (Eta²):** {eta2:.3f}")
        interp = f"**해석:** {_sig_text(p)}"
        if not np.isnan(eta2):
            interp += f" / 효과크기: {eta2:.2f} ({_eta2_level(eta2)})"
        result_md.append(interp)

        fig = px.box(groups, x=cat, y=num, points="all", color=cat)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("\n".join(result_md))
    return "\n".join(["### 그룹 간 평균 비교", *result_md]), None


def run_chi_square(df: pd.DataFrame, cat_cols: List[str]):
    st.subheader("🔢 카이제곱 독립성 검정")
    st.caption("두 범주형 변수의 독립성 검정. p<0.05면 연관성이 있으며, Cramer's V(0~1)로 강도를 해석합니다.")
    c1 = st.selectbox("범주형 변수 1", options=cat_cols, key="chi1")
    c2 = st.selectbox("범주형 변수 2", options=[c for c in cat_cols if c != c1], key="chi2")

    sub = df[[c1, c2]].dropna()
    ct = pd.crosstab(sub[c1], sub[c2])
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    st.write("**교차표**")
    st.dataframe(ct)

    cv = cramers_v(chi2, n=ct.values.sum(), r=ct.shape[0], c=ct.shape[1])

    md_base = textwrap.dedent(f"""
    ### 카이제곱 검정
    - 카이제곱 통계량: {chi2:.3f}
    - 자유도: {dof}
    - p-value: {p:.4g}
    - 효과크기 (Cramer's V): {cv:.3f}
    """)
    md = md_base + f"\n**해석:** {_sig_text(p)} / 연관성 강도: {_cramers_v_level(cv)}"
    st.markdown(md)

    fig = px.imshow(ct, text_auto=True, aspect="auto", labels=dict(x=c2, y=c1, color="빈도"))
    st.plotly_chart(fig, use_container_width=True)

    return md, ct


def run_linear_regression(df: pd.DataFrame, target: str, exclude_cols: List[str]):
    st.subheader("📐 선형 회귀")
    st.caption("RMSE는 예측 오차 크기, R²는 설명력(0~1). 낮은 RMSE, 높은 R²가 바람직합니다.")
    candidates = [c for c in df.columns if c != target and c not in exclude_cols]
    x_cols = st.multiselect("설명변수 선택", options=candidates, default=[c for c in candidates if pd.api.types.is_numeric_dtype(df[c])][:5])
    if not x_cols:
        st.info("설명변수를 하나 이상 선택하세요.")
        return "", None

    pre, feat_names = _build_preprocess(df, x_cols)

    data = df[x_cols + [target]].dropna()
    X = data[x_cols]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reg_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
    reg_pipe.fit(X_train, y_train)
    y_pred = reg_pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # 회귀분석 결과 대시보드
    st.markdown("### 📊 회귀분석 결과 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r2_badge = create_effect_size_badge(r2, "r_squared")
        display_metric_card("결정계수 (R²)", f"{r2:.3f}", r2_badge.split("**")[1].split("**")[0], 
                          "good" if r2 >= 0.7 else "warning" if r2 >= 0.5 else "error")
    
    with col2:
        try:
            y_sd = float(np.std(y_test.values, ddof=1))
            rel_rmse = rmse / y_sd if y_sd > 0 else np.nan
            if np.isfinite(rel_rmse):
                color = "good" if rel_rmse < 0.5 else "warning" if rel_rmse < 1.0 else "error"
                display_metric_card("상대 RMSE", f"{rel_rmse:.3f}", "표준편차 대비", color)
        except Exception:
            display_metric_card("RMSE", f"{rmse:.4g}", color="normal")
    
    with col3:
        mae = float(np.mean(np.abs(y_test - y_pred)))
        display_metric_card("평균 절대 오차", f"{mae:.3f}", color="normal")
    
    with col4:
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100) if np.all(y_test != 0) else np.nan
        if np.isfinite(mape):
            color = "good" if mape < 10 else "warning" if mape < 20 else "error"
            display_metric_card("MAPE", f"{mape:.1f}%", color=color)

    try:
        y_sd = float(np.std(y_test.values, ddof=1))
        rel_rmse = rmse / y_sd if y_sd > 0 else np.nan
    except Exception:
        rel_rmse = np.nan
    
    interp_lr = f"**해석:** 설명력 R²={r2:.2f} ({_r2_level(r2)})"
    if np.isfinite(rel_rmse):
        grade = "양호" if rel_rmse < 0.5 else ("보통" if rel_rmse < 1.0 else "낮음")
        interp_lr += f" / 상대 RMSE(표준편차 대비)={rel_rmse:.2f} → 예측 성능 {grade}"
    st.markdown(interp_lr)

    # 예측 vs 실제 그래프 (개선된 시각화)
    df_plot = pd.DataFrame({"실제": y_test.values, "예측": y_pred})
    fig = px.scatter(df_plot, x="실제", y="예측", trendline="ols", 
                     title="예측값 vs 실제값 (대각선에 가까울수록 좋음)")
    
    # 완벽한 예측 라인 추가 (y=x)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="red", dash="dash"), name="완벽한 예측")
    
    st.plotly_chart(fig, use_container_width=True)

    # 잔차 플롯
    residuals = y_test - y_pred
    fig_resid = px.scatter(x=y_pred, y=residuals, labels={"x": "예측값", "y": "잔차"},
                           title="잔차 플롯 (무작위 분포가 이상적)")
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)

    try:
        X_design = pre.transform(X_train)
        X_design = sm.add_constant(X_design)
        ols_model = sm.OLS(y_train, X_design).fit()
        
        with st.expander("📋 상세 통계 요약 보기"):
            st.text(ols_model.summary())
    except Exception as e:
        st.info(f"계수 요약 생성 중 스킵: {e}")

    md = textwrap.dedent(f"""
    ### 선형 회귀 결과
    - RMSE: {rmse:.4g}
    - R²: {r2:.4g}
    - 해석: 설명력 R²={r2:.2f} ({_r2_level(r2)}){f" / 상대 RMSE={rel_rmse:.2f}" if np.isfinite(rel_rmse) else ""}
    """)
    return md, None


def run_logistic_regression(df: pd.DataFrame, target: str, exclude_cols: List[str]):
    st.subheader("🧭 로지스틱 회귀 (분류)")
    st.caption("정확도/정밀도/재현율/F1은 분류 성능, ROC-AUC는 1에 가까울수록 좋습니다.")
    candidates = [c for c in df.columns if c != target and c not in exclude_cols]
    x_cols = st.multiselect("설명변수 선택", options=candidates, default=[c for c in candidates if c != target][:5])

    if not x_cols:
        st.info("설명변수를 하나 이상 선택하세요.")
        return "", None

    data = df[x_cols + [target]].dropna()
    X = data[x_cols]
    y = data[target]

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    pre, feat_names = _build_preprocess(df, x_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=500))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    st.markdown(f"**Accuracy:** {acc:.4g}  |  **Precision:** {pr:.4g}  |  **Recall:** {rc:.4g}  |  **F1:** {f1:.4g}")

    try:
        if len(np.unique(y_test)) == 2:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, thr = roc_curve(y_test, y_proba)
            df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            fig = px.line(df_roc, x="FPR", y="TPR")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**ROC AUC:** {auc:.4g}")
    except Exception:
        pass

    st.markdown("**분류 리포트**")
    st.text(classification_report(y_test, y_pred))

    try:
        auc_val = None
        if len(np.unique(y_test)) == 2:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc_val = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc_val = None
    interp_cls = f"**해석:** 정확도 {acc:.2f}, F1 {f1:.2f}"
    if auc_val is not None and np.isfinite(auc_val):
        interp_cls += f", AUC {auc_val:.2f} ({_auc_level(auc_val)})"
    st.markdown(interp_cls)

    md = textwrap.dedent(f"""
    ### 로지스틱 회귀 결과
    - Accuracy: {acc:.4g}
    - Precision/Recall/F1(가중): {pr:.4g}/{rc:.4g}/{f1:.4g}
    - 해석: 정확도 {acc:.2f}, F1 {f1:.2f}{f", AUC {auc_val:.2f} ({_auc_level(auc_val)})" if auc_val is not None and np.isfinite(auc_val) else ""}
    """)
    return md, None


def run_kmeans(df: pd.DataFrame, numeric_cols: List[str]):
    st.subheader("🧩 KMeans 군집화")
    st.caption("데이터를 k개의 군집으로 나눕니다. 실루엣 지수 0.5↑이면 군집 분리가 양호합니다.")
    cols = st.multiselect("군집에 사용할 연속형 변수", options=numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
    if len(cols) < 2:
        st.info("두 개 이상 선택하세요.")
        return "", None
    k = st.slider("군집 수 (k)", min_value=2, max_value=10, value=3, step=1)

    data = df[cols].dropna()
    if len(data) < k:
        st.warning("표본 수가 k보다 적습니다. 다른 k 또는 변수 선택.")
        return "", None

    X = StandardScaler().fit_transform(data.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels.astype(str)})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(X, labels))
        sil_text = f"{sil:.3f} (0.5↑ 좋음, 0~0.5 중간, <0 분리 약함)"
    except Exception:
        sil = np.nan
        sil_text = "계산 불가"

    md = textwrap.dedent(f"""
    ### KMeans 군집화
    - 선택 변수 수: {len(cols)}
    - k: {k}
    - 설명분산 (PCA, 2D): {pca.explained_variance_ratio_.sum():.3f}
    - 실루엣 계수: {sil_text}
    - 해석: 2D 설명분산이 낮으면 시각화에서 군집이 겹쳐 보일 수 있습니다. 실루엣이 높을수록 군집 분리가 선명합니다.
    """)
    return md, None


def run_time_series(df: pd.DataFrame, datetime_cols: List[str], numeric_cols: List[str]):
    st.subheader("🕒 시계열 분석")
    st.caption("시계열을 추세/계절/잔차로 분해하여 패턴을 확인합니다.")
    dcol = st.selectbox("날짜/시간 컬럼", options=datetime_cols)
    vcol = st.selectbox("분석할 연속형 변수", options=numeric_cols)
    freq = st.selectbox("리샘플 주기", options=["D", "W", "M"], format_func=lambda x: {"D": "일별", "W": "주별", "M": "월별"}[x])

    df2 = df[[dcol, vcol]].dropna().copy()
    if not np.issubdtype(df2[dcol].dtype, np.datetime64):
        df2[dcol] = pd.to_datetime(df2[dcol], errors="coerce")
    df2 = df2.dropna(subset=[dcol]).sort_values(dcol)
    ts = df2.set_index(dcol)[vcol].astype(float).resample(freq).mean().dropna()

    st.markdown("**추세 그래프**")
    fig = px.line(ts.reset_index(), x=dcol, y=vcol)
    st.plotly_chart(fig, use_container_width=True)

    md = ["### 시계열 분석"]
    try:
        period = {"D": 7, "W": 52, "M": 12}[freq]
        decomp = seasonal_decompose(ts, period=period, model="additive", extrapolate_trend='freq')
        comp_df = pd.DataFrame({
            "timestamp": ts.index,
            "관측치": ts.values,
            "추세": decomp.trend.values,
            "계절": decomp.seasonal.values,
            "잔차": decomp.resid.values,
        })
        for comp in ["추세", "계절", "잔차"]:
            fig_c = px.line(comp_df, x="timestamp", y=comp)
            st.plotly_chart(fig_c, use_container_width=True)
        try:
            obs_sd = float(np.nanstd(comp_df["관측치"]))
            seas_sd = float(np.nanstd(comp_df["계절"]))
            ratio = seas_sd / obs_sd if obs_sd > 0 else np.nan
            level = "뚜렷" if np.isfinite(ratio) and ratio >= 0.3 else "약함"
            md.append(f"계절성 강도(표준편차 비율): {ratio:.2f} → {level}")
        except Exception:
            pass
        md.append("시계열을 추세/계절/잔차로 분해했습니다.")
    except Exception as e:
        st.info(f"계절분해 스킵: {e}")

    return "\n".join(md), None


def run_insights(df: pd.DataFrame, it: InferredTypes):
    st.subheader("💡 자동 인사이트")
    
    # 전체 데이터 요약 대시보드
    st.markdown("### 📊 데이터 전체 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_missing = df.isna().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        color = "good" if missing_pct < 5 else "warning" if missing_pct < 15 else "error"
        display_metric_card("전체 결측률", f"{missing_pct:.1f}%", f"{total_missing:,}개", color)
    
    with col2:
        duplicates = df.duplicated().sum()
        dup_pct = (duplicates / len(df)) * 100 if len(df) > 0 else 0
        color = "good" if dup_pct < 1 else "warning" if dup_pct < 5 else "error"
        display_metric_card("중복 행", f"{dup_pct:.1f}%", f"{duplicates}개", color)
    
    with col3:
        display_metric_card("데이터 크기", f"{len(df):,}×{len(df.columns)}", color="normal")
    
    with col4:
        numeric_cols = len(it.numeric)
        cat_cols = len(it.categorical + it.boolean)
        display_metric_card("변수 구성", f"수치:{numeric_cols}, 범주:{cat_cols}", color="normal")

    parts: List[str] = ["### 자동 인사이트 요약"]

    # 결측 상위 (시각화 개선)
    miss = df.isna().mean().sort_values(ascending=False)
    miss_top = miss.head(10)[miss.head(10) > 0]
    if not miss_top.empty:
        st.markdown("### 🔍 결측값 분석")
        
        # 결측률 차트
        fig_missing = px.bar(x=miss_top.values, y=miss_top.index, orientation='h',
                           title="컬럼별 결측률", labels={"x": "결측률", "y": "컬럼"})
        fig_missing.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_missing, use_container_width=True)
        
        # 심각한 결측 경고
        severe_missing = miss_top[miss_top > 0.5]
        if not severe_missing.empty:
            st.error(f"🚨 **심각한 결측**: {len(severe_missing)}개 컬럼이 50% 이상 결측")
            for col, rate in severe_missing.items():
                st.markdown(f"- **{col}**: {rate:.1%} 결측")
        
        parts.append("**결측 상위**\n" + "\n".join([f"- {c}: {v:.1%}" for c, v in miss_top.head(5).items()]))

    # 상관 상위 (시각화 개선)
    if len(it.numeric) >= 2:
        st.markdown("### 🔗 강한 상관관계")
        corr = df[it.numeric].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False).head(5)
        
        if len(pairs) > 0:
            # 상관관계 차트
            corr_data = pd.DataFrame({
                '변수쌍': [f"{i} ↔ {j}" for (i, j), v in pairs.items()],
                '상관계수': pairs.values
            })
            
            fig_corr = px.bar(corr_data, x='상관계수', y='변수쌍', orientation='h',
                            title="강한 상관관계 Top 5", color='상관계수',
                            color_continuous_scale='Reds')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 다중공선성 경고
            strong_corr = pairs[pairs >= 0.8]
            if len(strong_corr) > 0:
                st.warning(f"⚠️ **다중공선성 주의**: {len(strong_corr)}개 쌍이 |r|≥0.8")
            
            parts.append("**강한 상관(상위 5)**\n" + "\n".join([f"- {i}–{j}: |r|={v:.2f}" for (i, j), v in pairs.items()]))

    # 집단차이 스캔 (시각화 개선)
    st.markdown("### 🧪 그룹 간 유의한 차이 탐지")
    hits: List[Tuple[float, str]] = []
    cats = (it.categorical + it.boolean)[:5]
    for cat in cats:
        levels = df[cat].dropna().unique()
        if 2 <= len(levels) <= 7:
            for num in it.numeric[:8]:
                sub = df[[num, cat]].dropna()
                if sub[cat].nunique() < 2:
                    continue
                groups = [sub[sub[cat] == lv][num].values for lv in sub[cat].unique()]
                if any(len(g) < 3 for g in groups):
                    continue
                try:
                    _, p = stats.f_oneway(*groups)
                    hits.append((p, f"{num} ~ {cat}"))
                except Exception:
                    continue
    
    hits = sorted(hits, key=lambda x: x[0])[:10]
    if hits:
        # 유의성 차트
        sig_data = pd.DataFrame({
            '변수쌍': [name for p, name in hits],
            'p-value': [p for p, name in hits],
            '유의성': ['매우 유의' if p < 0.001 else '유의' if p < 0.05 else '경계적' if p < 0.1 else '비유의' for p, name in hits]
        })
        
        fig_sig = px.bar(sig_data.head(8), x='p-value', y='변수쌍', orientation='h',
                        title="집단 간 차이 유의성 (ANOVA p-value)", color='유의성',
                        color_discrete_map={'매우 유의': 'darkgreen', '유의': 'green', 
                                          '경계적': 'orange', '비유의': 'red'})
        fig_sig.add_vline(x=0.05, line_dash="dash", line_color="red", 
                         annotation_text="p=0.05 (유의성 기준)")
        st.plotly_chart(fig_sig, use_container_width=True)
        
        # 매우 유의한 차이 강조
        very_sig = [name for p, name in hits if p < 0.001]
        if very_sig:
            st.success(f"🎯 **매우 유의한 차이 발견**: {len(very_sig)}개")
            for name in very_sig[:5]:
                st.markdown(f"- {name}")
        
        parts.append("**집단 간 차이 감지(ANOVA, 상위 5)**\n" + "\n".join([f"- {name}: {create_significance_badge(p)}" for p, name in hits[:5]]))

    # 이상치 간단 탐지
    if it.numeric:
        st.markdown("### 🚨 이상치 간단 탐지")
        outlier_summary = []
        
        for col in it.numeric[:5]:  # 상위 5개 수치 컬럼만
            s = df[col].dropna()
            if len(s) > 0:
                q1, q3 = np.percentile(s, [25, 75])
                iqr = q3 - q1
                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                outlier_pct = (outliers / len(s)) * 100
                outlier_summary.append({'컬럼': col, '이상치_개수': outliers, '이상치_비율': outlier_pct})
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            fig_outlier = px.bar(outlier_df, x='컬럼', y='이상치_비율',
                               title="컬럼별 이상치 비율 (IQR 기준)")
            st.plotly_chart(fig_outlier, use_container_width=True)

    return "\n".join(parts), None


def run_feature_importance(df: pd.DataFrame, target_hint: Optional[str]):
    st.subheader("🌟 특성 중요도")
    st.caption("퍼뮤테이션 중요도(검증셋)을 우선 사용하고, 실패 시 랜덤포레스트 중요도로 폴백합니다. 값이 클수록 타겟에 미치는 영향이 큽니다.")
    if target_hint is None or target_hint not in df.columns:
        tgt = st.selectbox("타겟 선택", options=df.columns.tolist())
    else:
        tgt = target_hint
        st.info(f"타겟: **{tgt}**")

    X_cols = [c for c in df.columns if c != tgt]
    X = df[X_cols]
    y = df[tgt]

    problem = "reg" if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 2 else "clf"

    pre, _ = _build_preprocess(df, X_cols)

    strat = None
    if problem == "clf":
        vc = pd.Series(y).value_counts()
        if len(vc) < 2:
            st.warning("타겟 클래스가 1개라 분류 중요도를 계산할 수 없습니다.")
            return "### 특성 중요도\n타겟 클래스가 1개라 계산 불가", None
        if vc.min() >= 2:
            strat = y
        else:
            st.info("타겟의 일부 클래스 표본 수가 2 미만이라 층화 없이 분할합니다. (희소 클래스 주의)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

    if problem == "reg":
        model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=300, random_state=42))])
    else:
        model = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
    model.fit(X_train, y_train)

    imp_df = pd.DataFrame()
    try:
        imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        try:
            feat_names = model.named_steps["pre"].get_feature_names_out().tolist()
        except Exception:
            feat_names = [f"f{i}" for i in range(len(imp.importances_mean))]
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp.importances_mean})
    except Exception:
        pass

    if imp_df.empty:
        try:
            rf = model.named_steps.get("rf")
            fi = getattr(rf, "feature_importances_", None)
            if fi is not None:
                try:
                    feat_names = model.named_steps["pre"].get_feature_names_out().tolist()
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(fi))]
                imp_df = pd.DataFrame({"feature": feat_names, "importance": fi})
        except Exception:
            pass

    if not imp_df.empty:
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        fig = px.bar(imp_df.iloc[:15][::-1], x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        md = "### 특성 중요도 (상위)\n" + "\n".join([f"- {r.feature}: {r.importance:.3f}" for r in imp_df.itertuples(index=False)])
    else:
        st.info("중요도를 계산하지 못했습니다. (데이터/버전에 따라 달라질 수 있음)")
        md = "### 특성 중요도\n계산 불가"
    return md, None


def run_tukey(df: pd.DataFrame, it: InferredTypes):
    st.subheader("🧪 사후검정 (Tukey HSD)")
    num = st.selectbox("연속형 변수", options=it.numeric)
    cat = st.selectbox("그룹(범주형) 변수", options=(it.categorical + it.boolean))
    sub = df[[num, cat]].dropna()
    if sub[cat].nunique() < 3:
        st.info("Tukey HSD는 3개 이상의 그룹에서 의미가 있습니다.")
        return "", None

    res = pairwise_tukeyhsd(sub[num].values, sub[cat].astype(str).values)

    # 표 변환
    try:
        tbl = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
        for c in ["meandiff", "p-adj", "lower", "upper"]:
            if c in tbl.columns:
                tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
        if "reject" in tbl.columns:
            tbl["reject"] = tbl["reject"].astype(str)
    except Exception:
        tbl = pd.DataFrame()

    n_pairs = len(tbl)
    sig_tbl = tbl[tbl.get("p-adj", pd.Series(dtype=float)) < 0.05] if not tbl.empty else pd.DataFrame()
    sig_cnt = int(len(sig_tbl)) if not sig_tbl.empty else 0

    MAX_SHOW = 200
    if not tbl.empty:
        sorted_tbl = tbl.sort_values("p-adj", na_position="last") if "p-adj" in tbl.columns else tbl
        st.markdown(f"**총 비교쌍:** {n_pairs:,}  |  **유의(p<0.05):** {sig_cnt:,} ({(sig_cnt/n_pairs):.1% if n_pairs>0 else 0})")
        top_preview = sorted_tbl.head(20)
        st.markdown("**상위 20개(가장 유의한 순)**")
        st.dataframe(top_preview, use_container_width=True)

        csv_bytes = sorted_tbl.to_csv(index=False).encode("utf-8-sig")
        st.download_button("전체 결과 CSV 다운로드", data=csv_bytes, file_name="tukey_results.csv", mime="text/csv")

        if n_pairs <= MAX_SHOW:
            with st.expander("전체 결과 표 보기"):
                st.dataframe(sorted_tbl, use_container_width=True)
    else:
        st.text(res.summary())

    means = sub.groupby(cat)[num].agg(["mean", "count", "std"]).reset_index()
    means["se"] = means["std"] / np.sqrt(means["count"].clip(lower=1))
    fig = px.bar(means, x=cat, y="mean", error_y="se")
    st.plotly_chart(fig, use_container_width=True)

    md_lines = [
        "### Tukey HSD 결과",
        f"총 비교쌍: {n_pairs:,}",
        f"유의한 쌍(p<0.05): {sig_cnt:,}",
    ]
    if sig_cnt > 0 and not sig_tbl.empty:
        top_lines = [f"- {r.group1} vs {r.group2}: p={r['p-adj']:.3g}" for _, r in sig_tbl.sort_values("p-adj").head(10).iterrows()]
        md_lines.append("**상위 10개 유의 쌍**")
        md_lines.extend(top_lines)
    md = "\n".join(md_lines)
    return md, None


def run_outliers(df: pd.DataFrame, it: InferredTypes):
    st.subheader("🚨 이상치 탐지")
    st.caption("IQR 경계(1.5×IQR)로 단변량 이상치를 보고, 옵션으로 Isolation Forest로 다변량 이상치를 찾습니다.")
    cols = st.multiselect("검사할 연속형 변수", options=it.numeric, default=it.numeric[:min(5, len(it.numeric))])
    if not cols:
        return "", None

    summary = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        count = int(mask.sum())
        summary.append({"컬럼": c, "IQR 범위": f"[{lo:.3g}, {hi:.3g}]", "이상치 수": count})
    if summary:
        st.markdown("**IQR 기준 요약**")
        st.dataframe(pd.DataFrame(summary))

    if st.checkbox("Isolation Forest로 다변량 이상치 탐지") and len(cols) >= 2:
        sub = df[cols].dropna()
        if len(sub) >= 10:
            X = StandardScaler().fit_transform(sub.values)
            iso = IsolationForest(random_state=42, contamination="auto")
            labels = iso.fit_predict(X)
            out_idx = sub.index[labels] == -1
            st.markdown(f"**Isolation Forest 이상치 수:** {(out_idx).sum()}")
            if (out_idx).sum() > 0:
                st.dataframe(df.loc[sub.index[out_idx]].head(50))
        else:
            st.info("표본 수가 적어 다변량 이상치 탐지를 생략합니다.")

    md = """### 이상치 탐지
- IQR로 컬럼별 경계와 개수를 요약했습니다. 다변량 이상치는 Isolation Forest 옵션으로 확인하세요."""
    return md, None


def run_regression_diagnostics(df: pd.DataFrame, target_hint: Optional[str]):
    st.subheader("🩺 회귀 진단")
    if target_hint is None or target_hint not in df.columns or not pd.api.types.is_numeric_dtype(df[target_hint]):
        st.info("연속형 타겟을 사이드바에서 지정하면 회귀 진단을 제공합니다.")
        return "", None

    tgt = target_hint
    num_cols = [c for c in df.columns if c != tgt and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        st.info("진단을 위한 연속형 설명변수가 2개 이상 필요합니다.")
        return "", None

    data = df[num_cols + [tgt]].dropna()
    X = data[num_cols]
    y = data[tgt]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    vif_df = pd.DataFrame({
        "변수": ["const"] + num_cols,
        "VIF": [np.nan] + [variance_inflation_factor(X_const.values, i + 1) for i in range(len(num_cols))],
    })
    st.markdown("**VIF (다중공선성)**")
    st.dataframe(vif_df)

    fitted = model.fittedvalues
    resid = model.resid

    fig1 = px.scatter(x=fitted, y=resid, labels={"x": "적합값", "y": "잔차"})
    st.plotly_chart(fig1, use_container_width=True)

    qq = sm.ProbPlot(resid)
    theo = qq.theoretical_quantiles
    samp = np.sort(resid)
    fig2 = px.scatter(x=theo, y=samp, labels={"x": "이론 분위수", "y": "표본 분위수"})
    st.plotly_chart(fig2, use_container_width=True)

    md = """### 회귀 진단
- VIF가 10을 크게 넘는 변수는 다중공선성 가능성 있음
- 잔차가 등분산/정규성을 크게 벗어나면 모델 수정 필요"""
    return md, None

# =============================================================
#  메인 UI
# =============================================================

st.title("📊 CSV 통계분석")
st.markdown("""
<div style="padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1rem;">
    <h4>🎯 이 도구의 특징</h4>
    <ul>
        <li><strong>자동 분석 제안</strong>: 데이터 타입을 자동으로 분석하여 적합한 통계분석을 추천합니다</li>
        <li><strong>직관적인 결과</strong>: 색상 코딩과 시각적 지표로 결과를 쉽게 이해할 수 있습니다</li>
        <li><strong>초보자 친화적</strong>: 복잡한 통계 용어를 쉬운 말로 설명합니다</li>
        <li><strong>종합 보고서</strong>: 모든 분석 결과를 정리된 보고서로 다운로드할 수 있습니다</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 도움말 확장 가능한 섹션
with st.expander("❓ 사용법 및 도움말"):
    st.markdown("""
    ### 📋 기본 사용법
    1. **CSV 파일 업로드**: 왼쪽 사이드바에서 CSV 파일을 선택하고 업로드하세요
    2. **타겟 변수 설정** (선택사항): 예측하고 싶은 변수가 있다면 타겟 변수로 설정하세요
    3. **분석 선택**: 추천된 분석 목록에서 원하는 분석을 선택하세요
    4. **결과 확인**: 색상 코딩된 결과와 해석을 확인하세요
    5. **보고서 다운로드**: 분석이 완료되면 전체 보고서를 다운로드하세요
    
    ### 🎨 결과 해석 가이드
    - **🟩 초록색**: 좋은 결과나 권장 상태
    - **🟡 노란색**: 주의가 필요한 상태  
    - **🔴 빨간색**: 문제가 있거나 개선이 필요한 상태
    - **📊 진행률 바**: 각종 지표의 수준을 시각적으로 표시
    
    ### 💡 분석 유형별 가이드
    - **기초 EDA**: 데이터의 기본 특성과 분포를 파악할 때 사용
    - **상관분석**: 변수 간의 관계를 확인할 때 사용
    - **그룹 비교**: 범주별로 평균에 차이가 있는지 확인할 때 사용
    - **회귀분석**: 특정 변수를 예측하고 싶을 때 사용
    - **자동 인사이트**: 데이터에서 자동으로 패턴을 찾고 싶을 때 사용
    """)

with st.sidebar:
    st.header("1) CSV 업로드")
    with st.form("load_form", clear_on_submit=False):
        up = st.file_uploader("CSV 파일 선택", type=["csv"], key="up")
        st.markdown("**고급 설정**")
        col1, col2 = st.columns(2)
        sep = col1.text_input("구분자 (비워두면 자동)", value=st.session_state.get("sep", ""))
        enc = col2.text_input("인코딩 (비워두면 자동)", value=st.session_state.get("enc", ""))

        st.divider()
        st.header("2) 타겟 변수 (선택)")
        target_hint_in = st.text_input("예: price, outcome 등 (없으면 비워두기)", value=st.session_state.get("target_hint", ""))
        submitted = st.form_submit_button("데이터 불러오기", type="primary")

    if submitted:
        if up is None:
            st.warning("CSV 파일을 선택하세요.")
        else:
            try:
                df_tmp = read_csv_safely(up, sep or None, enc or None)
                st.session_state["df"] = df_tmp
                st.session_state["sep"] = sep
                st.session_state["enc"] = enc
                st.session_state["target_hint"] = (target_hint_in or None)
                st.session_state["file_name"] = up.name
                st.success(f"파일 로드 완료! (행 {len(df_tmp)}, 열 {len(df_tmp.columns)})")
            except Exception as e:
                st.error(f"CSV를 읽는 중 오류: {e}")

    if "df" in st.session_state:
        st.caption(f"불러온 파일: {st.session_state.get('file_name','(이름 없음)')}")
        if st.button("🔁 다른 파일로 교체/옵션 변경", use_container_width=True):
            for k in ["df", "file_name"]:
                if k in st.session_state:
                    del st.session_state[k]

if "df" not in st.session_state:
    st.info("왼쪽 사이드바에서 CSV 파일과 옵션을 선택한 뒤 **데이터 불러오기** 버튼을 눌러주세요.")
    st.stop()

# 이후부터는 세션의 df와 타깃 힌트를 사용
df = st.session_state["df"]
target_hint = st.session_state.get("target_hint")

st.success(f"파일 로드 완료! (행 {len(df)}, 열 {len(df.columns)})")

# 타입 추론
inferred = infer_types(df)

# 개요
with st.expander("데이터 개요", expanded=True):
    st.write("**컬럼 타입 추론**")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("연속형", len(inferred.numeric))
    t2.metric("범주형", len(inferred.categorical))
    t3.metric("불리언", len(inferred.boolean))
    t4.metric("날짜/시간", len(inferred.datetime))

    st.dataframe(df.head())

    with st.expander("컬럼별 요약 보기"):
        try:
            desc = df.describe(include="all", datetime_is_numeric=True).T
        except TypeError:
            desc = df.describe(include="all").T
        st.dataframe(desc)

# 제안 생성
suggestions = suggest_analyses(df, inferred, target_hint)

st.header("🎯 가능한 분석 제안")

# 타겟 설정 상태에 따른 안내
if target_hint and target_hint in df.columns:
    st.success(f"✅ **타겟 변수 설정됨**: {target_hint}")
    st.info("🎯 타겟 변수가 설정되어 예측 모델링과 특성 중요도 분석이 가능합니다!")
else:
    st.warning("⚠️ **타겟 변수 미설정**: 예측 분석을 원한다면 사이드바에서 타겟 변수를 설정하세요.")
    with st.expander("💡 타겟 변수란?"):
        st.markdown("""
        **타겟 변수**는 예측하거나 설명하고 싶은 변수입니다.
        
        **예시:**
        - 고객 데이터에서 **'만족도'**를 예측하고 싶다면 → 타겟: 만족도
        - 주택 데이터에서 **'가격'**을 예측하고 싶다면 → 타겟: 가격  
        - 의료 데이터에서 **'질병여부'**를 예측하고 싶다면 → 타겟: 질병여부
        
        **타겟 설정 시 추가 분석:**
        - ✅ 회귀/분류 예측 모델
        - ✅ 특성 중요도 (어떤 변수가 타겟에 영향을 미치는지)
        - ✅ 회귀 진단 (모델의 품질 검증)
        """)

st.divider()

# 분석 제안 목록 (타겟 설정 여부에 따라 다르게 표시)
for s in suggestions:
    if s.label.startswith("✅"):
        st.success(f"**{s.label}** — {s.desc}")
    elif s.label.startswith("⚠️"):
        st.warning(f"**{s.label}** — {s.desc}")
    else:
        st.markdown(f"- **{s.label}** — {s.desc}")

st.divider()

# 실행할 분석 선택
labels_map = {s.label: s.key for s in suggestions}
# 기본 선택: 초보자 친화(EDA/상관/자동 인사이트/이상치, +특성중요도(있으면))
default_keys = {"eda", "correlation", "insights", "outliers"}
if any(s.key == "featimp" for s in suggestions):
    default_keys.add("featimp")
default_labels = [s.label for s in suggestions if s.key in default_keys]
chosen_labels = st.multiselect(
    "실행할 분석을 선택하세요 (여러 개 가능)",
    options=list(labels_map.keys()),
    default=default_labels,
)

# 세션 상태 관리
if "run" not in st.session_state:
    st.session_state.run = False
if "chosen_labels" not in st.session_state:
    st.session_state.chosen_labels = []

col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    run_click = st.button("분석 실행", type="primary")
with col_run2:
    reset_click = st.button("결과 초기화")

if run_click:
    st.session_state.run = True
    st.session_state.chosen_labels = chosen_labels

if reset_click:
    st.session_state.run = False
    st.session_state.chosen_labels = []

report_parts: List[str] = []

if st.session_state.run and st.session_state.chosen_labels:
    st.header("🎯 분석 결과")

    # 전체 결과 요약 섹션
    st.markdown("### 🌟 핵심 인사이트 요약")
    
    # 인사이트 카드들을 위한 컨테이너
    insight_container = st.container()
    key_insights = []
    
    active_labels = [lbl for lbl in st.session_state.chosen_labels if lbl in labels_map]
    
    # 간단한 사전 분석으로 핵심 인사이트 추출
    with insight_container:
        col1, col2, col3 = st.columns(3)
        
        # 데이터 품질 인사이트
        with col1:
            total_missing = df.isna().sum().sum()
            total_cells = len(df) * len(df.columns)
            missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            
            if missing_pct < 5:
                st.success("✅ **데이터 품질 우수**\n\n결측률이 5% 미만으로 분석에 적합합니다.")
            elif missing_pct < 15:
                st.warning("⚠️ **데이터 품질 보통**\n\n일부 결측값이 있어 주의가 필요합니다.")
            else:
                st.error("🚨 **데이터 품질 주의**\n\n결측률이 높아 전처리가 필요합니다.")
        
        # 변수 구성 인사이트
        with col2:
            num_vars = len(inferred.numeric)
            cat_vars = len(inferred.categorical + inferred.boolean)
            
            if num_vars >= 3 and cat_vars >= 2:
                st.success("🎯 **분석 가능성 높음**\n\n다양한 통계분석이 가능한 데이터입니다.")
            elif num_vars >= 2 or cat_vars >= 2:
                st.info("📊 **기본 분석 가능**\n\n기본적인 통계분석이 가능합니다.")
            else:
                st.warning("📈 **제한적 분석**\n\n변수가 적어 분석이 제한적입니다.")
        
        # 샘플 크기 인사이트
        with col3:
            sample_size = len(df)
            
            if sample_size >= 1000:
                st.success("📊 **충분한 샘플**\n\n통계적 검정에 적합한 크기입니다.")
            elif sample_size >= 100:
                st.info("📈 **적절한 샘플**\n\n기본 분석에 적합합니다.")
            else:
                st.warning("⚠️ **작은 샘플**\n\n결과 해석 시 주의가 필요합니다.")

    st.divider()

    for lbl in active_labels:
        key = labels_map[lbl]
        st.divider()
        if key == "eda":
            st.subheader("🔍 기초 탐색적 데이터 분석 (EDA)")
            
            # 컬럼 선택과 기본 정보
            col = st.selectbox("분석할 컬럼 선택", options=df.columns.tolist(), key=f"eda_{lbl}")
            s = df[col]
            
            # 기본 통계 대시보드
            st.markdown("### 📊 기본 통계 요약")
            col1, col2, col3, col4 = st.columns(4)
            
            miss = float(s.isna().mean())
            with col1:
                color = "good" if miss < 0.05 else "warning" if miss < 0.2 else "error"
                display_metric_card("결측률", f"{miss:.1%}", f"{s.isna().sum()}개", color)
            
            with col2:
                display_metric_card("총 개수", f"{len(s):,}", color="normal")
            
            with col3:
                unique_count = s.nunique(dropna=True)
                unique_pct = (unique_count / len(s.dropna())) * 100 if len(s.dropna()) > 0 else 0
                display_metric_card("고유값", f"{unique_count:,}", f"{unique_pct:.1f}%", "normal")
            
            with col4:
                data_type = "수치형" if pd.api.types.is_numeric_dtype(s) else "범주형"
                display_metric_card("데이터 타입", data_type, color="normal")
            
            show_distribution(df, col)
            
            if pd.api.types.is_numeric_dtype(s):
                # 수치형 변수 상세 분석
                st.markdown("### 📈 수치형 변수 상세 분석")
                
                s_clean = s.dropna()
                if len(s_clean) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 중심경향 지표**")
                        mean_val = float(s_clean.mean())
                        median_val = float(s_clean.median())
                        mode_val = float(s_clean.mode().iloc[0]) if len(s_clean.mode()) > 0 else np.nan
                        
                        st.metric("평균 (Mean)", f"{mean_val:.3f}")
                        st.metric("중앙값 (Median)", f"{median_val:.3f}")
                        if np.isfinite(mode_val):
                            st.metric("최빈값 (Mode)", f"{mode_val:.3f}")
                    
                    with col2:
                        st.markdown("**📏 분산 지표**")
                        std_val = float(s_clean.std())
                        var_val = float(s_clean.var())
                        cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
                        
                        st.metric("표준편차", f"{std_val:.3f}")
                        st.metric("분산", f"{var_val:.3f}")
                        if np.isfinite(cv_val):
                            st.metric("변동계수 (CV)", f"{cv_val:.1f}%")
                    
                    # 분포 모양 분석
                    skew = float(s_clean.skew())
                    kurt = float(s_clean.kurtosis())
                    
                    st.markdown("### 📐 분포의 모양")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if abs(skew) < 0.5:
                            skew_interp = "대칭적 분포 ⚖️"
                            skew_color = "good"
                        elif skew > 0:
                            skew_interp = "오른쪽 꼬리가 긴 분포 ↗️"
                            skew_color = "warning"
                        else:
                            skew_interp = "왼쪽 꼬리가 긴 분포 ↖️"
                            skew_color = "warning"
                        
                        display_metric_card("왜도 (Skewness)", f"{skew:.3f}", skew_interp, skew_color)
                    
                    with col2:
                        if abs(kurt) < 0.5:
                            kurt_interp = "정규분포와 비슷한 뾰족함 📊"
                            kurt_color = "good"
                        elif kurt > 0:
                            kurt_interp = "정규분포보다 뾰족함 📈"
                            kurt_color = "warning"
                        else:
                            kurt_interp = "정규분포보다 평평함 📉"
                            kurt_color = "warning"
                        
                        display_metric_card("첨도 (Kurtosis)", f"{kurt:.3f}", kurt_interp, kurt_color)
                    
                    # 이상치 탐지
                    q1, q3 = np.percentile(s_clean, [25, 75])
                    iqr = q3 - q1
                    outliers = ((s_clean < q1 - 1.5 * iqr) | (s_clean > q3 + 1.5 * iqr)).sum()
                    outlier_pct = (outliers / len(s_clean)) * 100
                    
                    if outliers > 0:
                        color = "warning" if outlier_pct < 5 else "error"
                        st.markdown("### 🚨 이상치 탐지")
                        display_metric_card("이상치 개수", f"{outliers}개", f"{outlier_pct:.1f}%", color)
                
                md = textwrap.dedent(f"""
                ### 기초 EDA 요약 - {col} (수치형)
                - 결측 비율: {miss:.1%}
                - 평균: {mean_val:.3f}, 중앙값: {median_val:.3f}
                - 왜도: {skew:.3f} ({skew_interp})
                - 이상치: {outliers}개 ({outlier_pct:.1f}%)
                """)
            else:
                # 범주형 변수 상세 분석
                st.markdown("### 📋 범주형 변수 상세 분석")
                
                nunq = int(s.nunique(dropna=True))
                value_counts = s.value_counts().head(10)
                
                # 빈도 차트
                if len(value_counts) > 0:
                    fig_freq = px.bar(x=value_counts.index.astype(str), y=value_counts.values,
                                    title=f"상위 {min(10, len(value_counts))}개 값의 빈도")
                    st.plotly_chart(fig_freq, use_container_width=True)
                
                # 분포 균등성 분석
                total_count = s.count()
                if total_count > 0:
                    entropy = -sum((p := count/total_count) * np.log2(p) for count in value_counts if count > 0)
                    max_entropy = np.log2(min(nunq, len(value_counts)))
                    balance_ratio = entropy / max_entropy if max_entropy > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        color = "good" if balance_ratio > 0.8 else "warning" if balance_ratio > 0.5 else "error"
                        display_metric_card("분포 균등성", f"{balance_ratio:.1%}", 
                                          "균등할수록 좋음" if balance_ratio > 0.8 else "불균등함", color)
                    
                    with col2:
                        most_common_pct = (value_counts.iloc[0] / total_count) * 100
                        color = "warning" if most_common_pct > 50 else "error" if most_common_pct > 80 else "good"
                        display_metric_card("최빈값 비율", f"{most_common_pct:.1f}%", 
                                          f"'{value_counts.index[0]}'", color)
                
                top3 = value_counts.head(3)
                top_lines = "\n".join([f"  • {idx}: {val}개 ({val/total_count:.1%})" for idx, val in top3.items()])
                
                md = textwrap.dedent(f"""
                ### 기초 EDA 요약 - {col} (범주형)
                - 고유값: {nunq}개
                - 결측 비율: {miss:.1%}
                - 분포 균등성: {balance_ratio:.1%}
                - 최빈값: '{value_counts.index[0]}' ({most_common_pct:.1f}%)

                **상위 3개 값**
                {top_lines}
                """)
            
            # 실용적인 조언 추가
            st.markdown("### 💡 데이터 품질 조언")
            advice = []
            
            if miss > 0.2:
                advice.append("⚠️ **결측률이 높습니다** - 결측값 처리 전략을 고려하세요")
            elif miss > 0.05:
                advice.append("📝 **결측값이 일부 있습니다** - 분석 시 주의가 필요합니다")
            
            if pd.api.types.is_numeric_dtype(s):
                if outlier_pct > 5:
                    advice.append("🚨 **이상치가 많습니다** - 이상치 제거나 변환을 고려하세요")
                if abs(skew) > 1:
                    advice.append("📐 **분포가 심하게 치우쳐 있습니다** - 로그변환 등을 고려하세요")
            else:
                if nunq == len(s.dropna()):
                    advice.append("🔍 **모든 값이 고유합니다** - 식별자 컬럼일 가능성이 높습니다")
                elif balance_ratio < 0.5:
                    advice.append("⚖️ **분포가 불균등합니다** - 클래스 불균형을 고려하세요")
            
            if advice:
                for adv in advice:
                    st.markdown(adv)
            else:
                st.success("✅ **데이터 품질이 양호합니다!**")
            
            st.markdown(md)
            report_parts.append(md)

        elif key == "correlation":
            md, _ = run_correlation(df, inferred.numeric)
            st.markdown(md)
            report_parts.append(md)

        elif key == "group_compare":
            md, _ = run_group_compare(df, inferred.numeric, inferred.categorical + inferred.boolean)
            report_parts.append(md)

        elif key == "chi_square":
            md, _ = run_chi_square(df, inferred.categorical + inferred.boolean)
            report_parts.append(md)

        elif key == "linreg":
            if target_hint and target_hint in df.columns and pd.api.types.is_numeric_dtype(df[target_hint]):
                # 타겟이 미리 설정된 경우
                st.info(f"🎯 **설정된 타겟**: {target_hint}")
                tgt = target_hint
            else:
                # 타겟이 없는 경우 사용자가 선택
                st.warning("⚠️ 타겟 변수가 설정되지 않았습니다. 분석할 타겟을 선택하세요.")
                tgt = st.selectbox("📊 예측하고 싶은 연속형 변수 선택", options=inferred.numeric)
                if tgt:
                    st.info(f"💡 **추천**: 다음번에는 사이드바에서 '{tgt}'를 타겟으로 미리 설정하세요!")
            
            md, _ = run_linear_regression(df, tgt, exclude_cols=[tgt])
            report_parts.append(md)

        elif key == "logreg":
            if target_hint and target_hint in df.columns and not pd.api.types.is_numeric_dtype(df[target_hint]):
                # 타겟이 미리 설정된 경우
                st.info(f"🎯 **설정된 타겟**: {target_hint}")
                tgt = target_hint
            else:
                # 타겟이 없는 경우 사용자가 선택
                st.warning("⚠️ 타겟 변수가 설정되지 않았습니다. 분석할 타겟을 선택하세요.")
                tgt = st.selectbox("📊 예측하고 싶은 범주형 변수 선택", options=inferred.categorical + inferred.boolean)
                if tgt:
                    st.info(f"💡 **추천**: 다음번에는 사이드바에서 '{tgt}'를 타겟으로 미리 설정하세요!")
            
            md, _ = run_logistic_regression(df, tgt, exclude_cols=[tgt])
            report_parts.append(md)

        elif key == "featimp":
            if target_hint and target_hint in df.columns:
                st.info(f"🎯 **설정된 타겟**: {target_hint}")
                md, _ = run_feature_importance(df, target_hint)
            else:
                st.warning("⚠️ 타겟 변수가 설정되지 않았습니다. 중요도를 분석할 타겟을 선택하세요.")
                md, _ = run_feature_importance(df, None)  # 함수 내에서 다시 선택하게 됨
                
            st.markdown(md)
            report_parts.append(md)

        elif key == "kmeans":
            md, _ = run_kmeans(df, inferred.numeric)
            st.markdown(md)
            report_parts.append(md)

        elif key == "timeseries":
            if not inferred.datetime:
                st.info("날짜/시간 컬럼이 없어 시계열 분석을 생략합니다.")
            else:
                md, _ = run_time_series(df, inferred.datetime, inferred.numeric)
                st.markdown(md)
                report_parts.append(md)

        elif key == "tukey":
            md, _ = run_tukey(df, inferred)
            st.markdown(md)
            report_parts.append(md)

        elif key == "outliers":
            md, _ = run_outliers(df, inferred)
            st.markdown(md)
            report_parts.append(md)

        elif key == "insights":
            md, _ = run_insights(df, inferred)
            st.markdown(md)
            report_parts.append(md)

    st.divider()

    # ---------- 최종 요약 및 권고사항 ----------
    st.markdown("### 🎯 분석 완료 요약")
    
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("완료된 분석", f"{len(active_labels)}개", "선택한 분석 모두 완료")
    
    with summary_cols[1]:
        analyzed_rows = len(df)
        st.metric("분석된 데이터", f"{analyzed_rows:,}행", f"{len(df.columns)}개 컬럼")
    
    with summary_cols[2]:
        if "insights" in [labels_map[lbl] for lbl in active_labels]:
            st.metric("핵심 인사이트", "포함됨", "자동 인사이트 분석 완료")
        else:
            st.metric("추가 분석", "가능", "더 많은 인사이트 발견 가능")

    # 다음 단계 권장사항
    st.markdown("### 💡 다음 단계 권장사항")
    
    next_steps = []
    completed_keys = {labels_map[lbl] for lbl in active_labels}
    
    if "correlation" in completed_keys and "regdiag" not in completed_keys:
        next_steps.append("🔍 **회귀 진단 실행** - 상관관계를 발견했다면 회귀 진단으로 다중공선성을 확인해보세요")
    
    if "group_compare" in completed_keys and "tukey" not in completed_keys:
        next_steps.append("📊 **사후검정 실행** - 그룹 간 차이가 유의하다면 Tukey 검정으로 어떤 그룹끼리 다른지 확인해보세요")
    
    if len(inferred.numeric) >= 2 and "kmeans" not in completed_keys:
        next_steps.append("🧩 **군집분석 고려** - 데이터에 숨겨진 패턴을 찾기 위해 K-means 분석을 해보세요")
    
    if target_hint and "featimp" not in completed_keys:
        next_steps.append("🌟 **특성 중요도 분석** - 타겟 변수에 영향을 미치는 주요 변수들을 확인해보세요")
    
    if "outliers" not in completed_keys:
        next_steps.append("🚨 **이상치 탐지** - 데이터 품질 향상을 위해 이상치를 확인해보세요")

    if next_steps:
        for step in next_steps:
            st.markdown(step)
    else:
        st.success("🎉 **분석 완료!** 모든 주요 분석을 마쳤습니다. 결과를 바탕으로 데이터 기반 의사결정을 진행하세요.")

    # ---------- 리포트 모드 ---------- 
    st.divider()
    st.markdown("### 📋 분석 보고서")
    
    # 보고서 헤더 추가
    report_header = f"""
# 📊 데이터 분석 보고서

**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**데이터 크기**: {len(df):,}행 × {len(df.columns)}개 컬럼
**실행된 분석**: {', '.join(active_labels)}

---

"""
    
    report_md = report_header + "\n\n".join(report_parts)
    
    # 보고서 미리보기와 다운로드
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.expander("📝 전체 보고서 미리보기", expanded=False):
            st.markdown(report_md)
    
    with col2:
        st.download_button(
            "📥 보고서 다운로드\n(Markdown)",
            data=report_md.encode("utf-8-sig"),
            file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        # 간단한 통계 요약도 다운로드 옵션 제공
        if not df.empty:
            summary_stats = df.describe(include='all').to_csv()
            st.download_button(
                "📊 기초통계 다운로드\n(CSV)",
                data=summary_stats.encode("utf-8-sig"),
                file_name=f"summary_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

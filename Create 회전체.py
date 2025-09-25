# app.py
"""
Streamlit app: 회전체의 성질 탐구
사용법:
  1) 이 파일을 app.py로 저장
  2) 필요한 패키지 설치: pip install streamlit plotly numpy
  3) 실행: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="회전체 Explorer", layout="wide")

st.title("🔁 회전체 (Solid of Revolution) 탐구 앱")
st.markdown(
    """
    이 앱은 함수 \(y=f(x)\)를 \(x\)-축을 중심으로 회전시켜 생성되는 회전체의 특성을 수치적으로 계산하고 시각화합니다.
    - 입력 함수는 파이썬 표현식(예: `0.5*(x-1)**2 + 0.2`, `np.sin(x) + 1`) 으로 넣어주세요.
    - 내부 반지름(inner radius)을 주면 속이 빈 회전체(워셔)를 계산합니다.
    """
)

# --- 사이드바: 사용자 입력 ---
st.sidebar.header("설정")
st.sidebar.markdown("함수와 구간을 입력하세요. 독립변수는 `x` 입니다.")

func_text = st.sidebar.text_input("외부 곡선 y = f(x)", value="np.sin(x) + 1.5")
inner_text = st.sidebar.text_input("내부 곡선 y = g(x) (속이 없으면 0)", value="0.0")
x0 = st.sidebar.number_input("구간 시작 x0", value=0.0, format="%.4f")
x1 = st.sidebar.number_input("구간 끝 x1", value=2 * np.pi, format="%.4f")
if x1 <= x0:
    st.sidebar.error("구간 끝(x1)은 시작(x0)보다 커야 합니다.")
samples = st.sidebar.slider("샘플 수 (정밀도)", min_value=200, max_value=5000, value=1000, step=100)
density = st.sidebar.number_input("밀도 ρ (질량밀도)", value=1.0, format="%.6f")
show_surface = st.sidebar.checkbox("3D 회전체 시각화 보이기", value=True)
show_profile = st.sidebar.checkbox("원단면(디스크)·미분 등 보이기", value=True)
show_slices = st.sidebar.checkbox("얇은 조각(slices) 표시", value=False)

# 안전한 eval 환경 구성
safe_locals = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
               "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "abs": np.abs}

# 함수 평가 (벡터화)
def make_func(expr: str):
    # 반환: vectorized function f(x)
    def f(x):
        return eval(expr, {"__builtins__": {}}, {**safe_locals, "x": x})
    return np.vectorize(f)

try:
    f_outer = make_func(func_text)
    f_inner = make_func(inner_text)
    # 샘플링
    x = np.linspace(x0, x1, samples)
    y = f_outer(x).astype(float)
    y_inner = f_inner(x).astype(float)
    # 강제: 반지름이 음수가 되면 0으로 치환 (물리적으로)
    y = np.maximum(y, 0.0)
    y_inner = np.maximum(y_inner, 0.0)
    # 내부 반지름이 외부보다 큰 경우 경고 표시(수치적으로는 절댓값 차로 처리)
    if np.any(y_inner > y):
        st.warning("경고: 어떤 구간에서 내부반지름(inner) > 외부반지름(outer) 입니다. 그 구간은 0으로 처리됩니다.")
        # 교정
        y_inner = np.minimum(y_inner, y)
except Exception as e:
    st.error(f"함수 평가 에러: {e}")
    st.stop()

# 수치 미분 (중심차분 근사)
dy_dx = np.gradient(y, x)

# 수치 적분 보조
def trapz(x_vals, y_vals):
    return np.trapz(y_vals, x_vals)

# --- 계산: 부피, 겉넓이, 관성모멘트, 체적 중심 ---
# 부피 (washer): V = π ∫ (R^2 - r^2) dx
area_integrand = np.pi * (y**2 - y_inner**2)
V = trapz(x, area_integrand)  # 체적

# 표면적 (회전체 겉넓이): S = 2π ∫ R * sqrt(1 + (dR/dx)^2) dx
# 단, 내부 표면적도 포함하려면 내부함수 부분을 더해준다 (속빈 경우 내부면적 포함)
S_outer_integrand = 2 * np.pi * y * np.sqrt(1 + dy_dx**2)
# 내부 면적 계산시 내부의 미분 필요:
dyin_dx = np.gradient(y_inner, x)
S_inner_integrand = 2 * np.pi * y_inner * np.sqrt(1 + dyin_dx**2)
# 단, 내부 반지름이 0이면 integrand 역시 0이므로 안전
S = trapz(x, S_outer_integrand - S_inner_integrand)  # 외부에서 내부 면적을 빼는 형태

# 관성모멘트(회전축 x-axis 기준) : 원판 모델 -> dI = (1/2) * dm * R^2, dm = ρ π R^2 dx
# 그래서 I = (1/2) * ρ * π ∫ (R^4 - r^4) dx
I_integrand = 0.5 * density * np.pi * (y**4 - y_inner**4)
I_x = trapz(x, I_integrand)

# 질량: m = ρ * V
mass = density * V

# 체적 중심 x-coordinate: x_c = (1/V) ∫ x dV, dV = π (R^2 - r^2) dx
xV_integrand = x * np.pi * (y**2 - y_inner**2)
x_center = trapz(x, xV_integrand) / V if V != 0 else np.nan

# --- 출력: 수치 결과 ---
st.subheader("계산 결과 (수치)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("부피 V", f"{V:.6g} (단위^3)")
col2.metric("겉넓이 S", f"{S:.6g} (단위^2)")
col3.metric("질량 m (ρ={:.3g})".format(density), f"{mass:.6g}")
col4.metric("관성모멘트 I_x", f"{I_x:.6g} (단위^5)")

st.write(f"체적 중심 (x 좌표): {x_center:.6g}")

# --- 그래프 영역 ---
st.subheader("시각화")

# 1) 2D 프로파일
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=y, mode="lines", name="외부 R(x)"))
fig2.add_trace(go.Scatter(x=x, y=y_inner, mode="lines", name="내부 r(x)"))
if show_profile:
    fig2.add_trace(go.Scatter(x=x, y=dy_dx, mode="lines", name="dR/dx (근사)", yaxis="y2",
                              line=dict(dash="dot")))
    # 두 번째 y축을 추가
    fig2.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="dR/dx")
    )
fig2.update_layout(title="곡선 프로파일 (y = f(x) 및 내부)", xaxis_title="x", yaxis_title="y")
st.plotly_chart(fig2, use_container_width=True)

# 2) 3D 회전체 시각화
if show_surface:
    # 회전체 표면을 만들기 위해 (x, theta) 격자 생성
    theta = np.linspace(0, 2 * np.pi, 120)
    X, Theta = np.meshgrid(x, theta)  # shape (len(theta), len(x))
    R_outer = np.tile(y, (len(theta), 1))
    R_inner = np.tile(y_inner, (len(theta), 1))
    Y_surf = R_outer * np.cos(Theta)
    Z_surf = R_outer * np.sin(Theta)
    # 외피
    surf_outer = go.Surface(x=X, y=Y_surf, z=Z_surf, opacity=0.9, name="외부 표면",
                            showscale=False)
    data = [surf_outer]
    # 내부가 있는 경우 내부 표면(반전된 법선처럼 보이도록) 추가
    if np.any(y_inner > 0):
        Y_in = R_inner * np.cos(Theta)
        Z_in = R_inner * np.sin(Theta)
        surf_inner = go.Surface(x=X, y=Y_in, z=Z_in, opacity=0.8, name="내부 표면", showscale=False)
        data.append(surf_inner)

    layout = go.Layout(
        title="회전체 3D 시각화 (x축이 회전축)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="auto"),
        width=1000,
        height=600
    )
    fig3 = go.Figure(data=data, layout=layout)

    # optional: thin slices to show disks
    if show_slices:
        n_slices = 20
        idx = np.linspace(0, len(x) - 1, n_slices).astype(int)
        for i in idx:
            xi = x[i]
            Ri = y[i]
            ri = y_inner[i]
            th = np.linspace(0, 2 * np.pi, 80)
            xs = np.full_like(th, xi)
            ys = Ri * np.cos(th)
            zs = Ri * np.sin(th)
            fig3.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=2), showlegend=False))
            if ri > 0:
                ys2 = ri * np.cos(th)
                zs2 = ri * np.sin(th)
                fig3.add_trace(go.Scatter3d(x=xs, y=ys2, z=zs2, mode="lines", line=dict(width=2), showlegend=False))

    st.plotly_chart(fig3, use_container_width=True)

# --- 상세 수식 / 설명 섹션 ---
with st.expander("수식 및 참고"):
    st.markdown(
        """
        - 부피 (washer): \( V = \pi \int_{x_0}^{x_1} \big(R(x)^2 - r(x)^2\big)\,dx \)
        - 겉넓이: \( S = 2\pi \int_{x_0}^{x_1} R(x)\sqrt{1 + (R'(x))^2}\,dx \) (내부면이 있으면 내부면적을 빼줍니다)
        - 관성모멘트 (x축 기준, 원판 근사): \( I_x = \tfrac{1}{2}\rho\pi \int_{x_0}^{x_1} \big(R(x)^4 - r(x)^4\big)\,dx \)
        - 체적 중심 (x 좌표): \( x_c = \dfrac{1}{V}\int_{x_0}^{x_1} x\, \pi\big(R(x)^2 - r(x)^2\big)\,dx \)
        """
    )
    st.markdown("**주의:** 수치적으로 미분/적분을 근사하고 있으므로, 샘플 수(samples)를 늘리면 결과가 더 정확해집니다.")

# --- 다운로드: 결과 CSV (옵션) ---
if st.button("샘플 데이터 다운로드 (CSV)"):
    import pandas as pd
    df = pd.DataFrame({
        "x": x,
        "R(x)": y,
        "r(x)": y_inner,
        "dR/dx": dy_dx
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("CSV 저장", data=csv, file_name="revolution_profile.csv", mime="text/csv")

st.markdown("---")
st.caption("작성: Streamlit 앱 — 회전체의 부피/면적/관성모멘트 계산 및 3D 시각화")

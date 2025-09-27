import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="중1 수학 탐구 어플", layout="wide")

# ----------------------------
# 사이드바 메뉴
# ----------------------------
st.sidebar.title("탐구 주제 선택")
menu = st.sidebar.radio(
    "아래 중 하나를 선택하세요.",
    (
        "다면체 성질 탐구하기",
        "회전체 탐구",
        "회전체 단면 보기",
        "나만의 회전체 만들기"
    )
)

# ----------------------------
# 1. 다면체 성질 탐구하기
# ----------------------------
if menu == "다면체 성질 탐구하기":
    st.header("📐 다면체 성질 탐구하기")

    solid_type = st.radio("도형 종류 선택", ["n각기둥", "n각뿔", "n각뿔대"])
    n = st.number_input("밑면의 변의 수 (n)", min_value=3, step=1)

    if solid_type == "n각기둥":
        faces = n + 2
        vertices = 2 * n
        edges = 3 * n
    elif solid_type == "n각뿔":
        faces = n + 1
        vertices = n + 1
        edges = 2 * n
    else:  # n각뿔대
        faces = n + 2
        vertices = 2 * n
        edges = 3 * n

    st.write(f"- **면의 수:** {faces}")
    st.write(f"- **꼭짓점의 수:** {vertices}")
    st.write(f"- **모서리의 수:** {edges}")

    if solid_type != "n각뿔대":
        st.latex("면 + 꼭짓점 - 모서리 = 2")
        st.write(f"검산: {faces} + {vertices} - {edges} = {faces + vertices - edges}")

    # 3D 시각화
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    base_x = np.cos(theta)
    base_y = np.sin(theta)

    if solid_type == "n각기둥":
        z_bottom = np.zeros_like(base_x)
        z_top = np.ones_like(base_x)
        fig = go.Figure(data=[
            go.Mesh3d(x=np.concatenate([base_x, base_x]),
                      y=np.concatenate([base_y, base_y]),
                      z=np.concatenate([z_bottom, z_top]),
                      alphahull=0, opacity=0.5, color="lightblue")
        ])
    elif solid_type == "n각뿔":
        apex = [0, 0, 1]
        fig = go.Figure(data=[
            go.Mesh3d(x=np.append(base_x, apex[0]),
                      y=np.append(base_y, apex[1]),
                      z=np.append(np.zeros_like(base_x), apex[2]),
                      alphahull=0, opacity=0.5, color="lightgreen")
        ])
    else:  # n각뿔대
        r1, r2 = 1.0, 0.5
        base_x1, base_y1 = r1*np.cos(theta), r1*np.sin(theta)
        base_x2, base_y2 = r2*np.cos(theta), r2*np.sin(theta)
        fig = go.Figure(data=[
            go.Mesh3d(x=np.concatenate([base_x1, base_x2]),
                      y=np.concatenate([base_y1, base_y2]),
                      z=np.concatenate([np.zeros_like(base_x1), np.ones_like(base_x2)]),
                      alphahull=0, opacity=0.5, color="orange")
        ])

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    st.subheader("3D 시각화")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 2. 회전체 탐구
# ----------------------------
elif menu == "회전체 탐구":
    st.header("회전체의 성질")
    rotation_shapes = {
        "원기둥": "밑면이 원인 직사각형을 회전해 만든 입체도형",
        "원뿔": "직각삼각형을 한 축을 기준으로 회전해 만든 입체도형",
        "구": "반원을 회전해 만든 입체도형",
        "원뿔대": "밑면의 반지름이 서로 다른 두 원 사이를 잇는 회전체",
    }
    shape = st.selectbox("회전체를 선택하세요.", list(rotation_shapes.keys()))
    st.write(f"### {shape}")
    st.write(f"**정의:** {rotation_shapes[shape]}")

    if shape == "원기둥":
        r = st.number_input("반지름 r", min_value=1.0, step=1.0)
        h = st.number_input("높이 h", min_value=1.0, step=1.0)
        st.write(f"- 부피 = πr²h = {np.pi*r**2*h:.2f}")
    elif shape == "원뿔":
        r = st.number_input("반지름 r", min_value=1.0, step=1.0)
        h = st.number_input("높이 h", min_value=1.0, step=1.0)
        st.write(f"- 부피 = (1/3)πr²h = {(1/3)*np.pi*r**2*h:.2f}")
    elif shape == "구":
        r = st.number_input("반지름 r", min_value=1.0, step=1.0)
        st.write(f"- 부피 = (4/3)πr³ = {(4/3)*np.pi*r**3:.2f}")
    else:
        r1 = st.number_input("밑면 반지름 r1", min_value=1.0, step=1.0)
        r2 = st.number_input("윗면 반지름 r2", min_value=1.0, step=1.0)
        h = st.number_input("높이 h", min_value=1.0, step=1.0)
        st.write(f"- 부피 = (1/3)πh(r1² + r2² + r1r2) = {(1/3)*np.pi*h*(r1**2 + r2**2 + r1*r2):.2f}")

# ----------------------------
# 3. 회전체 단면 보기
# ----------------------------
elif menu == "회전체 단면 보기":
    st.header("회전체 단면 관찰")
    shape = st.selectbox("도형 선택", ["원기둥", "원뿔"])
    cut_dir = st.radio("자르는 방향", ["수평 (밑면과 평행)", "수직 (축과 평행)"])

    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.axis("off")

    if shape == "원기둥":
        if cut_dir == "수평 (밑면과 평행)":
            circle = plt.Circle((0.5, 0.5), 0.4, color='orange')
            ax.add_patch(circle)
            st.write("단면: 원(circle)")
        else:
            rect = plt.Rectangle((0.1,0.1), 0.8, 0.8, color='lightblue')
            ax.add_patch(rect)
            st.write("단면: 직사각형(rectangle)")
    else:  # 원뿔
        if cut_dir == "수평 (밑면과 평행)":
            circle = plt.Circle((0.5, 0.5), 0.3, color='salmon')
            ax.add_patch(circle)
            st.write("단면: 원(circle)")
        else:
            triangle = np.array([[0.5,0.1], [0.1,0.9], [0.9,0.9], [0.5,0.1]])
            ax.plot(triangle[:,0], triangle[:,1], 'k')
            ax.fill(triangle[:,0], triangle[:,1], 'yellow')
            st.write("단면: 이등변삼각형(isosceles triangle)")

    st.pyplot(fig)

# ----------------------------
# 4. 나만의 회전체 만들기
# ----------------------------
elif menu == "나만의 회전체 만들기":
    st.header("🎨 나만의 회전체 만들기")
    st.write("왼쪽 캔버스에 단면 도형을 그리고, y축을 기준으로 회전시켜 회전체를 만들어 보세요.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas_custom",
    )

    if canvas_result.image_data is not None:
        img = np.mean(canvas_result.image_data[:, :, :3], axis=2)
        ys, xs = np.where(img < 200)

        if len(xs) > 0:
            xs = (xs - xs.min()) / (xs.max() - xs.min())
            ys = (ys - ys.min()) / (ys.max() - ys.min())

            theta = np.linspace(0, 2*np.pi, 60)
            Xs, Thetas = np.meshgrid(xs, theta)
            Ys, _ = np.meshgrid(ys, theta)

            Zs = Xs * np.cos(Thetas)
            Xs = Xs * np.sin(Thetas)

            fig = go.Figure(data=[go.Surface(
                x=Xs, y=Ys, z=Zs, colorscale="Viridis", opacity=0.7
            )])
            fig.update_layout(scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ))
            st.subheader("🌀 생성된 회전체")
            st.plotly_chart(fig, use_container_width=True)

            # 단면 탐구 기능
            st.subheader("✂️ 단면 탐구하기")
            cut_dir = st.radio("자르는 방향", ["수평 (밑면과 평행)", "수직 (축과 평행)"])

            if cut_dir == "수평 (밑면과 평행)":
                cut_height = st.slider("단면 높이 선택 (0~1)", 0.0, 1.0, 0.5, 0.05)
                fig_cut = go.Figure(data=[go.Scatter(
                    x=xs*np.cos(theta), y=xs*np.sin(theta), mode="markers"
                )])
                fig_cut.update_layout(title=f"높이 {cut_height:.2f}에서의 단면",
                                      xaxis=dict(visible=False),
                                      yaxis=dict(visible=False))
                st.plotly_chart(fig_cut, use_container_width=True)

            else:  # 수직 절단
                cut_pos = st.slider("절단 위치 (x축 기준, 0~1)", 0.0, 1.0, 0.5, 0.05)
                vertical_section = ys
                fig_cut = go.Figure(data=[go.Scatter(
                    x=vertical_section, y=xs, mode="markers"
                )])
                fig_cut.update_layout(title=f"x={cut_pos:.2f}에서의 수직 단면",
                                      xaxis=dict(visible=False),
                                      yaxis=dict(visible=False))
                st.plotly_chart(fig_cut, use_container_width=True)

        else:
            st.info("✏️ 먼저 캔버스에 도형을 그려주세요.")

st.markdown("---")
st.write("© 2025 중1 수학 탐구 어플 - Streamlit Demo")

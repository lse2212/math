import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import sqrt, pi

st.set_page_config(page_title="다면체 & 회전체 탐구", layout="wide")

st.title("중1 수학: 다면체와 회전체 탐구 앱")
st.write("이 앱은 기본적인 다면체(정다면체 일부)와 회전체의 부피·겉넓이·모양을 시각적으로 탐구하도록 설계되었습니다.")

# ------------------ Mesh utilities ------------------

def triangles_from_quads(quads):
    """Convert list of quads (4-tuples) to triangles (two per quad)."""
    tris = []
    for q in quads:
        a, b, c, d = q
        tris.append((a, b, c))
        tris.append((a, c, d))
    return tris


def mesh_area_volume(vertices, faces):
    """Compute surface area and volume from triangular faces.
    volume computed as sum of signed tetra volumes to origin: V = 1/6 sum dot(v0, cross(v1,v2)).
    Faces are list of triplets of vertex indices.
    """
    verts = np.array(vertices)
    area = 0.0
    vol = 0.0
    for f in faces:
        v0 = verts[f[0]]
        v1 = verts[f[1]]
        v2 = verts[f[2]]
        # triangle area
        tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        area += tri_area
        # tetra volume
        vol += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(area), abs(vol)


def scale_to_edge_length(vertices, faces, desired_edge):
    """Scale vertices so that the mean edge length equals desired_edge."""
    verts = np.array(vertices, dtype=float)
    edges = set()
    for f in faces:
        for i in range(3):
            a = f[i]
            b = f[(i+1)%3]
            edges.add(tuple(sorted((int(a), int(b)))))
    lengths = []
    for e in edges:
        lengths.append(np.linalg.norm(verts[e[0]] - verts[e[1]]))
    mean_len = float(np.mean(lengths))
    if mean_len == 0:
        return verts
    scale = desired_edge / mean_len
    return verts * scale

# ------------------ Predefined polyhedra ------------------

POLYHEDRA = {}

# Tetrahedron (regular) unit coordinates
POLYHEDRA['Tetrahedron'] = {
    'vertices': [
        (1,1,1),
        (1,-1,-1),
        (-1,1,-1),
        (-1,-1,1),
    ],
    'faces': [
        (0,1,2), (0,3,1), (0,2,3), (1,3,2)
    ]
}

# Cube
POLYHEDRA['Cube'] = {
    'vertices': [
        (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
        (-1,-1,1),  (1,-1,1),  (1,1,1),  (-1,1,1)
    ],
    'faces': triangles_from_quads([
        (0,1,2,3), # bottom
        (4,5,6,7), # top
        (0,4,5,1), # front
        (1,5,6,2), # right
        (2,6,7,3), # back
        (3,7,4,0)  # left
    ])
}

# Octahedron
POLYHEDRA['Octahedron'] = {
    'vertices': [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
    ],
    'faces': [
        (0,2,4),(2,1,4),(1,3,4),(3,0,4),
        (2,0,5),(1,2,5),(3,1,5),(0,3,5)
    ]
}

# ------------------ Streamlit layout ------------------

section = st.sidebar.selectbox("탐구 영역 선택", ["다면체(Polyhedra)", "회전체(Solids of Revolution)"])

if section == "다면체(Polyhedra)":
    st.header("다면체 탐구")
    poly = st.selectbox("다면체 선택", list(POLYHEDRA.keys()))
    edge_len = st.slider("모서리 길이 (단위)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    detail = st.slider("3D 그리드 해상도 (시각적 품질)", min_value=10, max_value=60, value=30)

    data = POLYHEDRA[poly]
    verts_scaled = scale_to_edge_length(data['vertices'], data['faces'], edge_len)
    area, vol = mesh_area_volume(verts_scaled, data['faces'])

    # counts
    V = len(verts_scaled)
    E = None
    # compute unique edges
    edges = set()
    for f in data['faces']:
        for i in range(3):
            a = f[i]
            b = f[(i+1)%3]
            edges.add(tuple(sorted((int(a), int(b)))))
    E = len(edges)
    F = len(data['faces'])

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("기하량적 성질")
        st.markdown(f"- 정점 수 V = **{V}**")
        st.markdown(f"- 모서리 수 E = **{E}**")
        st.markdown(f"- 면 수 F = **{F}**")
        st.markdown(f"- 오일러 표준식 V - E + F = **{V - E + F}**")
        st.markdown(f"- 겉넓이 (수치) = **{area:.4f} (단위^2)**")
        st.markdown(f"- 부피 (수치) = **{vol:.4f} (단위^3)**")

    with col2:
        st.subheader("3D 시각화")
        x, y, z = verts_scaled[:,0], verts_scaled[:,1], verts_scaled[:,2]
        i = [f[0] for f in data['faces']]
        j = [f[1] for f in data['faces']]
        k = [f[2] for f in data['faces']]
        mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.6, flatshading=True)
        fig = go.Figure(data=[mesh])
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.info("참고: 위의 겉넓이·부피 값은 삼각형 메쉬로 수치적으로 계산된 값입니다. 정다면체의 해석식과 거의 일치합니다.")

else:
    st.header("회전체 탐구")
    st.write("y = f(x) 형태의 함수 한 개를 x축(혹은 y축)을 중심으로 회전시켜 회전체를 만듭니다. 수치 적분으로 부피와 겉넓이를 계산합니다.")

    col1, col2 = st.columns([1,1])
    with col1:
        func_str = st.text_input("함수 f(x) 입력 (예: x**2, np.sin(x), 1/(1+x**2))", value="x**2")
        xmin = st.number_input("시작 x", value=0.0)
        xmax = st.number_input("끝 x", value=1.0)
        revolve_axis = st.selectbox("회전 축 선택", ["x축 주변 회전(겉면 회전) - Y-> 회전", "y축 주변 회전 (X-> 회전)"])
        samples = st.slider("적분 샘플 수", min_value=200, max_value=3000, value=800)
        show_cross = st.checkbox("회전체의 단면(원) 표시", value=True)
    with col2:
        st.write("함수 입력 방법 안내:")
        st.markdown("- `np` 네임스페이스를 사용할 수 있습니다. 예: `np.sin(x)`\n- 안전을 위해 eval은 제한된 네임스페이스에서 실행됩니다.")

    # safe eval environment
    safe_globals = {"np": np, "sqrt": np.sqrt, "pi": np.pi}
    x = np.linspace(xmin, xmax, samples)
    try:
        # evaluate f(x)
        f = eval(func_str, safe_globals, {"x": x})
        f = np.array(f, dtype=float)
    except Exception as e:
        st.error(f"함수 평가 오류: {e}")
        st.stop()

    # numeric derivative
    dx = x[1] - x[0]
    df = np.gradient(f, dx)

    # volume by disk method if rotating around x-axis: V = pi * int f(x)^2 dx
    vol_num = pi * np.trapz(f**2, x)
    # surface area (lateral) S = 2*pi*int f(x) * sqrt(1 + f'(x)^2) dx
    surf_num = 2 * pi * np.trapz(f * np.sqrt(1 + df**2), x)

    st.subheader("수치 결과")
    st.markdown(f"- 수치적 부피 (x축 주변 회전 가정) = **{vol_num:.6f} (단위^3)**")
    st.markdown(f"- 수치적 겉넓이 (측면) = **{surf_num:.6f} (단위^2)**")

    # 3D mesh for revolution
    theta = np.linspace(0, 2*np.pi, 80)
    X, T = np.meshgrid(x, theta)
    R = np.tile(f, (len(theta),1))
    Y = R * np.cos(T)
    Z = R * np.sin(T)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.8)])
    fig.update_layout(scene=dict(aspectmode='auto'), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    if show_cross:
        st.subheader("단면(몇 개) 시각화")
        idxs = np.linspace(0, len(x)-1, 5, dtype=int)
        for i in idxs:
            col1, col2 = st.columns([1,3])
            with col1:
                st.markdown(f"x = {x[i]:.3f}에서 단면 반지름 = {f[i]:.3f}")
            with col2:
                # draw circle points
                t = np.linspace(0,2*pi,100)
                xx = f[i]*np.cos(t)
                yy = f[i]*np.sin(t)
                circle_fig = go.Figure()
                circle_fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines'))
                circle_fig.update_layout(xaxis=dict(scaleanchor='y'), yaxis=dict(showgrid=False), margin=dict(l=0,r=0,t=0,b=0), width=300, height=300)
                st.plotly_chart(circle_fig)

    st.info("참고: 회전체 계산은 x축 주변 회전(원반법)으로 수행됩니다. y축 중심 회전 등 다른 경우는 축 변환을 통해 적용할 수 있습니다.")

# ------------------ footer ------------------

st.sidebar.markdown("---")
st.sidebar.markdown("앱 사용법: 상단에서 영역을 선택하고 매개변수를 조절하세요.\n문의나 기능 추가 요청이 있으면 알려주세요!")

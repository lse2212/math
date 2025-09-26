import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("중1 수학 - 다면체와 회전체의 성질 탐구 어플")

st.sidebar.header("탐구 주제 선택")
menu = st.sidebar.radio(
   "아래 중 하나를 선택하세요.",
     ("다면체 탐구", "다면체 전개도", "회전체 탐구", "회전체 단면 보기")
 )
 
 polyhedrons = {
     "정사면체": {"면": 4, "모서리": 6, "꼭짓점": 4},
     "정육면체(큐브)": {"면": 6, "모서리": 12, "꼭짓점": 8},
     "정팔면체": {"면": 8, "모서리": 12, "꼭짓점": 6},
     "정십이면체": {"면": 12, "모서리": 30, "꼭짓점": 20},
     "정이십면체": {"면": 20, "모서리": 30, "꼭짓점": 12},
 }
 
 rotation_shapes = {
     "원기둥": "밑면이 원인 직육면체를 회전해 만든 입체도형",
     "원뿔": "직각삼각형을 한 축을 기준으로 회전해 만든 입체도형",
     "구": "반원을 회전해 만든 입체도형"
 }
 
 if menu == "다면체 탐구":
     st.header('다면체의 성질')
     poly_name = st.selectbox("다면체를 선택하세요.", list(polyhedrons.keys()))
     st.write(f"### {poly_name}의 성질")
     poly = polyhedrons[poly_name]
     st.write(f"- **면의 수:** {poly['면']}개")
     st.write(f"- **모서리의 수:** {poly['모서리']}개")
     st.write(f"- **꼭짓점의 수:** {poly['꼭짓점']}개")
     st.latex("면 + 꼭짓점 - 모서리 = 2")
     if st.button("오일러의 정리 확인"):
         result = poly['면'] + poly['꼭짓점'] - poly['모서리']
        st.write(f"확인: {poly['면']} + {poly['꼭짓점']} - {poly['모서리']} = {result}")

elif menu == "다면체 전개도":
    st.header('다면체 전개도')
    poly_name = st.selectbox("전개도를 보고 싶은 도형을 선택하세요.", ["정육면체", "정사면체"])
    st.write(f"#### {poly_name}의 전개도")

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.axis('off')
    if poly_name == "정육면체":
        squares = [(1, 1), (2, 1), (3, 1), (2, 2), (2, 3), (2, 0)]
        for (x, y) in squares:
            rect = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='skyblue', linewidth=2)
            ax.add_patch(rect)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 4.5)
    else:
        triangles = [
            np.array([[0,0], [1,0], [0.5,0.866], [0,0]]), # base
            np.array([[0,0], [-0.5,-0.866], [0.5,0], [0,0]]), # left
            np.array([[1,0], [1.5,-0.866], [0.5,0], [1,0]]), # right
            np.array([[0.5,0.866],[0.5,1.732], [1,0.866], [0.5,0.866]]) # top
        ]
        for t in triangles:
            ax.plot(t[:,0]+1, t[:,1]+1, 'k', linewidth=2)
            ax.fill(t[:,0]+1, t[:,1]+1, 'lightgreen', alpha=0.8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 3)
    st.pyplot(fig)
    st.info("전개도를 직접 그리고, 각 면이 어떻게 이어지는지 관찰해보세요!")

elif menu == "회전체 탐구":
    st.header('회전체의 성질')
    shape = st.selectbox("회전체를 선택하세요.", list(rotation_shapes.keys()))
    st.write(f"### {shape}")
    st.write(f"**정의:** {rotation_shapes[shape]}")
    st.image(
        f"https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/{'Cylindre' if shape=='원기둥' else ('Cone' if shape=='원뿔' else 'Sphere')}_3D.svg/240px-{ 'Cylindre' if shape=='원기둥' else ('Cone' if shape=='원뿔' else 'Sphere') }_3D.svg.png",
        caption=f"{shape} 예시"
    )
    if shape == "원기둥":
        st.write("밑면의 넓이는 πr², 옆면의 넓이는 2πrh, 부피는 πr²h 입니다.")
    elif shape == "원뿔":
        st.write("밑면의 넓이는 πr², 옆면의 넓이는 πrl, 부피는 (1/3)πr²h 입니다.")
    else:
        st.write("겉넓이: 4πr², 부피: (4/3)πr³")

else:
    st.header("회전체 단면")
    st.write("회전체를 수평, 수직으로 잘랐을 때 단면의 모양을 확인할 수 있습니다.")
    shape = st.selectbox("도형을 선택하세요.", ["원기둥", "원뿔"])
    cut = st.radio("자르는 방향을 선택하세요.", ("수평 (밑면과 평행)", "수직 (축과 평행)"))

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.axis('off')
    if shape == "원기둥":
        if cut == "수평 (밑면과 평행)":
            circle = plt.Circle((0.5, 0.5), 0.4, color='orange', fill=True)
            ax.add_patch(circle)
            st.write("단면: **원(circle)**")
        else:
            rect = plt.Rectangle((0.1,0.1), 0.8, 0.8, color='lightblue')
            ax.add_patch(rect)
            st.write("단면: **직사각형(rectangle)**")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    else:
        if cut == "수평 (밑면과 평행)":
            ellipse = plt.Ellipse((0.5, 0.3), 0.7, 0.2, color='salmon', fill=True)
            ax.add_patch(ellipse)
            st.write("단면(아래쪽): **원(circle)**, 단면(중간): **작은 원(circle)**")
        else:
            triangle = np.array([[0.5,0.1], [0.1,0.9], [0.9,0.9], [0.5,0.1]])
            ax.plot(triangle[:,0], triangle[:,1], 'k')
            ax.fill(triangle[:,0], triangle[:,1], 'yellow', alpha=0.8)
            st.write("단면: **이등변삼각형(isosceles triangle)**")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    st.pyplot(fig)
    st.warning("회전체의 단면 모양을 직접 상상하거나 그려 보세요!")

st.markdown("---")
st.caption("🚀 이 앱은 Streamlit으로 제작되었습니다. 자유롭게 개선해서 사용하세요!")




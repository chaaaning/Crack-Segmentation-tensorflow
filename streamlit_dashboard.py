import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
# from utils.utils import load_image
# from bokeh.io import show, output_file
# from bokeh.plotting import figure
from pydeck.types import String

st.title('Crack Detection Result Dashboard')

RESULT_PATH = "./image_predictions/masking_result.csv"
DATA_PATH = r"D:\data\dj_data\대전시_도로_영상_객체_인식_데이터셋_2020_위치정보"        # 이건 이미지데이터 로드하려고 했는데 마스킹 결과는 굳이 대시보드에서 안보여주는 걸로

# -- Merge Image
@st.cache
def merge_img(frame, pred, img_size=(384,288), is_BGR=True):
    re_frm = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    if is_BGR:
        add_mask = cv2.cvtColor(re_frm, cv2.COLOR_BGR2RGB)
    else:
        add_mask = re_frm.copy()
    add_mask[pred[:,:]!=0]=[0,255,0]
    return add_mask

# -- Loading Data & Header
@st.cache
def load_data(path_name, s=None):
    if s==None:
        data = pd.read_csv(path_name)
    else:
        data = pd.read_csv(path_name, sep=s, engine='python',
                           names=["libarary", "version"])
    return data

# -- str to list
@st.cache
def str2list(str_arr):
    try:
        return list(map(int, str_arr.lstrip("[").rstrip("]").replace(" ","").split(",")))
    except:
        return []
@st.cache
def mat2csrmat(mask_indices, mask_indptr, crop_height=288, crop_width=384):
    try:
        test = csr_matrix((np.array([1]*len(mask_indices)), mask_indices, mask_indptr), shape=(crop_height, crop_width)).toarray()
    except:
        test = np.array([[0]*crop_width for _ in range(crop_height)])
    return test

data_load_state = st.text('Loading data...')
use_cols = ["path", "latitude", "longitude", "mask_indices", "mask_indptr", "masking_rate"]
show_cols = ["latitude", "longitude", "masking_rate"]
data = load_data(RESULT_PATH)[use_cols]
data_load_state.text("Done! (using st.cache)")

st.header('Model Environment')


c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Training")
    st.write("CPU : Intel 12th 12600KF")
    st.write("VGA : Nvidia RTX 3060 D6 12GB")
    st.write("RAM : Samsung DDR4-25600 32GB(Dual)")
    st.write("Python Version 3.9.13")
    # if st.checkbox('View Requirement train'):
    #     st.write(load_data(os.path.join(os.getcwd(), 'requirements.txt'), s="=="))
with c2:
    st.subheader("Inference")
    st.write("CPU : Intel 12th 12600K")
    st.write("VGA : Nvidia RTX 3070Ti D6x 8GB")
    st.write("RAM : Samsung DDR4-25600 32GB(Dual)")
    st.write("Python Version 3.9.13")
    # if st.checkbox('View Requirement test'):
    #     st.write(load_data(os.path.join(os.getcwd(), 'requirements_inference.txt'), s="=="))
with c3:
    st.subheader('Requirement')
    st.write(load_data(os.path.join(os.getcwd(), 'requirements.txt'), s="=="))

# -- Split Columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Location & Masking Ratio Table Data Ordered")
    show_data = data.copy()
    show_data["masking_rate"] = show_data["masking_rate"].apply(lambda x: round(x, 3))
    st.dataframe(data[show_cols].sort_values(by=["masking_rate"], ascending=False))
with col2:
    st.subheader('Masking Rate Histogram')
    hist_values = np.histogram(data["masking_rate"])[0]
    st.bar_chart(hist_values)
    
chart1, chart2 = st.columns(2)
with chart1:
    st.subheader("Seaborn Histplot")
    fig1, ax1 = plt.subplots()
    sns.histplot(ax=ax1, data=show_data["masking_rate"], kde=True)
    st.pyplot(fig1)
with chart2:
    st.subheader("Seaborn Boxplot")
    fig2, ax2 = plt.subplots()
    sns.boxplot(ax=ax2, data=show_data["masking_rate"])
    st.pyplot(fig2)


# -- Draw Maps

map_view_data = data[show_cols].copy()
map_view_data.dropna(axis=0, subset=['latitude', 'longitude'], inplace=True)
map_view_data['latitude'] = pd.to_numeric(map_view_data['latitude'])
map_view_data['longitude'] = pd.to_numeric(map_view_data['longitude'])
mean_crack_ratio = map_view_data.groupby(by=["latitude", "longitude"], as_index=False)["masking_rate"].mean()
mean_crack_ratio["stand_mr"] = mean_crack_ratio["masking_rate"]/mean_crack_ratio["masking_rate"].max()
# st.map(mean_crack_ratio)
center = [127.380, 36.360]
st.header("Draw Heatmap")
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        longitude=center[0],
        latitude=center[1],
        zoom=11,
        pitch=50,
        bearing=-15,
    ),
    tooltip={"html":"<b>Longitude: </b> {longitude} <br /> "
                    "<b>Latitude: </b>{latitude} <br /> "
                    "<b>Masking Rate: </b>{masking_rate}"},
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            map_view_data,
            get_position='[longitude, latitude]',
            get_radius=20,
            get_fill_color='[255, 255-255*masking_rate, 255-255*masking_rate, 255*masking_rate]',
            pickable=True,
            auto_highlight=True,    # 마우스 오버시 출력
            # get_radius="500*stand_mr"
            # get_elevation='stand_mr',
            # elevation_scale=4,
            # elevation_rate = [0, 1000]
        ),
        pdk.Layer(
            'HeatmapLayer',
            mean_crack_ratio,
            opacity=0.9,
            get_position=['longitude', 'latitude'],
            aggregation=String('MEAN'),
            get_weight = "stand_mr",
        )
    ],
))
st.header("Grid Chart")
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        longitude=center[0],
        latitude=center[1],
        zoom=11,
        pitch=50,
        bearing=-15,
    ),
    layers=[
        pdk.Layer(
            'GPUGridLayer',
            map_view_data,
            get_position='[longitude, latitude]',
            pickable=True,
            # auto_highlight=True,
            extruded=True,
            elevation_scale=3
        )
    ]
))


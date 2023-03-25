import io
import streamlit as st
import numpy as np
from src.utils import utils
import PIL.Image as Image
from src.reconstruct_image_from_representation import reconstruct_image_from_representation
from src.neural_style_transfer import neural_style_transfer

st.set_page_config(
    page_title="Neural Style Transfer Video Generation of image reconstruction",
    page_icon="\u2712",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Neural Style Transfer Video Generation")

# Sidebar
st.sidebar.header("Neural Style Transfer Video Generation")
with st.sidebar.expander('About the app'):
    st.write("""
        Use this application to play with the Neural Style Transfer
        by generating video of optimizer
    """)

# Reconstruct or Transfer
with st.sidebar.container():
    st.sidebar.subheader("Reconstruct or Transfer batao")

    Type = st.sidebar.selectbox("Do you want to reconstruct or transfer",
                                ["Reconstruct", "Transfer"])
    utils.yamlSet('type', Type)

# Optimizer
with st.sidebar.container():
    st.sidebar.subheader("Optimizer")

    optimizer = st.sidebar.selectbox("Choose Optimizer", ["Adam", "LBFGS"])
    utils.yamlSet('optimizer', optimizer)

    iterations = st.sidebar.slider("Iterations", 10, 3000)
    utils.yamlSet('iterations', iterations)

    if optimizer == "Adam":
        learning_rate = st.sidebar.slider("Learning Rate (100\u03BB)", 0.01,
                                          90.0)
        utils.yamlSet('learning_rate', learning_rate)
        st.sidebar.write("\u03BB = ", learning_rate / 100.0)

# Reconstruction
if Type == "Reconstruct":
    with st.sidebar.container():
        st.sidebar.subheader("Reconstruction")
        reconstruct = st.sidebar.selectbox("Reconstruct which image",
                                           ('Content', 'Style'))
        utils.yamlSet('reconstruct', reconstruct)

# Visualization
with st.sidebar.container():
    st.sidebar.subheader("Visualization")
    visualize = st.sidebar.selectbox(
        "Do you want to visualize feature maps of reconstruct images",
        ("Yes", "No"))
    utils.yamlSet('visualize', visualize)

# Model
with st.sidebar.container():
    st.sidebar.subheader("Model")
    model = st.sidebar.selectbox("Choose Model",
                                 ("VGG16", "VGG16-Experimental"))
    utils.yamlSet('model', model)

# # use layer
# if model == "VGG19":
#     with st.sidebar.container():
#         st.sidebar.subheader("Layer Type")
#         use = st.sidebar.selectbox("Which type of layer you want to use",
#                                ("convolution", "relu"))

# Init Image
if Type == "Transfer":
    with st.sidebar.container():
        st.sidebar.subheader("Init Image")
        initImage = st.sidebar.selectbox(
            "Init Image",
            ('Gaussian Noise Image', 'White Noise Image', 'Content', 'Style'))
        utils.yamlSet('initImage', initImage)

# Content Layer
with st.sidebar.container():
    st.sidebar.subheader("Content Layer")
    if model == "VGG16-Experimental":
        contentLayer = st.sidebar.selectbox(
            "Content Layer", ('relu1_1', 'relu2_1', 'relu2_2', 'relu3_1',
                              'relu3_2', 'relu4_1', 'relu4_3', 'relu5_1'))
    elif model == "VGG16":
        contentLayer = st.sidebar.selectbox(
            "Content Layer", ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'))
    utils.yamlSet('contentLayer', contentLayer)
    # elif model == "VGG19" and use == "relu":
    #     st.sidebar.selectbox("Content Layer",
    #                      ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'))
    # elif model == "VGG19" and use == "convolution":
    #     st.sidebar.selectbox("Content Layer",
    #                      ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2',
    #             'conv5_1'))

# Height
with st.sidebar.container():
    st.sidebar.subheader("Height")
    height = st.sidebar.slider("Height", 100, 6000, 400)
    utils.yamlSet('height', height)

# Representation saving frequency
with st.sidebar.container():
    st.sidebar.subheader("Representation Saving Frequency")
    reprSavFreq = st.sidebar.slider(
        "After how many iterations you want to save representation for "
        "video generation", 1, 100)
    utils.yamlSet('reprSavFreq', reprSavFreq)

if Type == "Transfer":
    # Content Weight
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        contentWeight = st.slider("Content Weight (1000\u03B1)", 0.01, 1000.0)
        utils.yamlSet('contentWeight', contentWeight)

    with col2:
        st.write("\u03B1 = ", contentWeight / 1000.0)

    # Style Weight
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        styleWeight = st.slider("Style Weight (1000\u03B2)", 0.01, 1000.0)
        utils.yamlSet('styleWeight', styleWeight)

    with col2:
        st.write("\u03B2 = ", styleWeight / 1000.0)

    # Total Variation Weight
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        totalVariationWeight = st.slider("Total Variation Weight (1000\u03B3)",
                                         0.01, 1000.0)
        utils.yamlSet('totalVariationWeight', totalVariationWeight)

    with col2:
        st.write("\u03B3 = ", totalVariationWeight / 1000.0)

# File upload
col1, col2 = st.columns([0.5, 0.5])
with col1:
    contentImage = st.file_uploader('Choose Content Image', type=['jpg'])
    if contentImage:
        st.image(contentImage)
        contentNumpy = np.asarray(
            Image.open(io.BytesIO(contentImage.getvalue())))
        contentPath = utils.save_numpy_array_as_jpg(contentNumpy, "content")
        utils.yamlSet('contentPath', contentPath)

with col2:
    styleImage = st.file_uploader('Choose Style Image', type=['jpg'])
    if styleImage:
        st.image(styleImage)
        styleNumpy = np.asarray(Image.open(io.BytesIO(styleImage.getvalue())))
        stylePath = utils.save_numpy_array_as_jpg(styleNumpy, "style")
        utils.yamlSet("stylePath", stylePath)

submit = st.button("Submit")

if submit:
    utils.clearDir()
    if Type == "Reconstruct":
        reconstruct_image_from_representation()
    elif Type == "Transfer":
        neural_style_transfer()
        video_file = open("src/data/transfer/out.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

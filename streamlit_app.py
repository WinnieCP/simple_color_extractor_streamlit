
import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

def generate_colors(image, k):
    resultion = 200
    kmeans = KMeans(n_clusters=k)
    np_img = np.array(image)
    dims = np.shape(image)
    idx_largest_dim = np.argmax(dims)
    if dims[idx_largest_dim] > resultion:
        sample = int(dims[idx_largest_dim]/resultion)
        np_img = np_img[::sample,::sample]
    colors = np_img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1],3))
    s = kmeans.fit(colors)
    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    colors = [rgb_to_hex(c) for c in centroid]
    count, color_index = np.histogram(labels,bins=np.arange(0,k+1),density=True)
    return np_img, count, colors

def rgb_to_hex(color_tuple):
    color_tuple = np.array(color_tuple).astype(int)
    r, g, b = color_tuple
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def extract_colors (image):
    img, counts, colors = generate_colors(image, k)
    dims = np.shape(img)
    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios':(3,1), 'wspace' : .1})
    ax[1].imshow(img)
    left = 0
    for count, color in zip(counts, colors):
        ax[0].barh(0, count, color=color, left=left)
        left += count
    for i in [0,1]:
        ax[i].axis('off')
    fig.set_size_inches(10,2)
    st.pyplot(fig)
    df = pd.DataFrame({'color':colors, 'percentage of image':counts})
    st.table(df)

if __name__ == '__main__':
    st.title("Main Colors of an Image")
    k = st.slider('How many colors would you like to extract?',min_value=1, max_value=15, value=5)

    url = st.text_input('Please provide the url to an image...', value="")

    if url:
        res = requests.get(url)
        image_from_url = Image.open(BytesIO(res.content)).convert('RGB')

    if st.button("Extract Colors from URL!", type="primary"):      
        extract_colors(image_from_url)

    img_file_buffer = st.file_uploader('...OR upload an image.')

    if img_file_buffer:
        image_from_file = Image.open(img_file_buffer).convert('RGB')

    if st.button("Extract Colors from File!", type="primary"):      
        extract_colors(image_from_file)
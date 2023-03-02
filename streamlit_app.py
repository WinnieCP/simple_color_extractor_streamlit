
import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import streamlit.components.v1 as components


def cluster_colors(image, k):
    # make sure the image is a numpy array
    np_img = np.array(image)
    # downsample image im one axis is longer than resolution
    resultion = 200
    dims = np.shape(image)
    idx_largest_dim = np.argmax(dims)
    if dims[idx_largest_dim] > resultion:
        sample = int(dims[idx_largest_dim]/resultion)
        np_img = np_img[::sample,::sample]
    # reshape the image to have dimensions [x, 3]
    colors = np_img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1],3))
    # determine k clusters in rgb-space
    kmeans = KMeans(n_clusters=k)
    _ = kmeans.fit(colors)
    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    # turn rgb colors into hex codes
    colors = [rgb_to_hex(c) for c in centroid]
    # determine the relative frequency of each color in the image
    count, color_index = np.histogram(labels,bins=np.arange(0,k+1),density=True)
    # determine a readable font color (black/white) for displaying the result later
    fc = np.sum(centroid, axis=-1) > 500
    font_color_dict = {1:'#000000',0:'#ffffff'}
    fontcolors = [font_color_dict[i] for i in fc]    
    return np_img, count, colors, fontcolors

def generate_figure(img, colors, counts):
    # generate a stacked bar plot of main colors
    # and down-sampled image display    
    dims = np.shape(img)
    fig, ax = plt.subplots(1, 2, 
                        sharey=True, 
                        gridspec_kw={'width_ratios':(3,1),
                                        'wspace' : .1})
    ax[1].imshow(img)
    left = 0
    for count, color in zip(counts, colors):
        ax[0].barh(0, count, color=color, left=left, height=dims[0],
                    align='edge', edgecolor='none')
        left += count
    for i in [0,1]:
        ax[i].set_ylim(dims[0],0)
        ax[i].axis('off')
    ax[0].set_xlim(0,1)
    fig.set_size_inches(10,2)
    fig.set_facecolor('0.95')
    return fig

def rgb_to_hex(color_tuple):
    # convert rgb to hex string
    color_tuple = np.array(color_tuple).astype(int)
    r, g, b = color_tuple
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def extract_colors (image):
    # get clustered main colors
    img, counts, colors, fontcolors = cluster_colors(image, k)
    # display the result as colored paragraphs
    st.write(f'Your image has the following {k} main colors:')
    html_string = ''.join ([f'<p style="background-color:{colors[i]}; text-align: center;'
                            + f'height: 25px; width: {100/min(7,k)}%; float: left; color:{fontcolors[i]};'
                            + f'font-family: Helvetica; line-height:25px; margin-top: 0; margin-bottom: 0;">{colors[i]}</p>' for i in range(k)]) 
    components.html(html_string)

    # display the results as a stacked bar plot + the downsampled image
    st.write('The image is composed of these colors as follows:')
    fig = generate_figure(img, colors, counts)
    st.pyplot(fig)
   
if __name__ == '__main__':
    
    img_url_example = 'https://images.thalia.media/-/BF2000-2000/3a12a709bdbe43cf8d82521ab66a5257/my-neighbor-totoro-picture-book-gebundene-ausgabe-hayao-miyazaki-englisch.jpeg'
    
    st.title("Image Color Extractor")
    st.subheader('How many colors?')

    k = st.slider('Choose how many colors you would like to extract.', min_value=1, max_value=14, value=7)
    st.subheader('Provide an image url...')
    url = st.text_input('Try this example or use your own link:', 
                        value=img_url_example)

    if url:
        res = requests.get(url)
        image_from_url = Image.open(BytesIO(res.content)).convert('RGB')
        if st.button("Extract Colors from URL!", type="primary"):      
            extract_colors(image_from_url)

    st.subheader('...OR upload an image.')
    img_file_buffer = st.file_uploader('')

    if img_file_buffer:
        image_from_file = Image.open(img_file_buffer).convert('RGB')
        if st.button("Extract Colors from File!", type="primary"):      
            extract_colors(image_from_file)
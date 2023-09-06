import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
from PIL import Image

# import os

import numpy as np
# import plotly.graph_objects as go

from utilities import *

# Global Variables
original_image_height  = 1920
original_image_width = 1441

scaled_left = 0
scaled_top = 0
scaled_right = 0
scaled_bottom = 0

# last_subdirectory = ''


def main():
    ary_box = np.zeros([1000,1000])
    st.title("Automobile Dent App")

    projectName = st.sidebar.text_input("Enter Project Name", "Project1")

    uploaded_file = st.sidebar.file_uploader("Choose a image file e.g. 'imageName.png'", type=["png","jpg"])
    if uploaded_file is not None:

        # print(uploaded_file)
        
        try:
            image = Image.open(uploaded_file)
            im = np.array(image)
            # if im has four channel i.e. including alpha channel, then remove the fourth channel
            # if im is gray scale then it has only 2 channels
            if len(im.shape) > 2 and im.shape[2] == 4:
                im = im[:, :, :3]
            # print(f'im shape: {im.shape}')

            original_image_height  = im.shape[0]
            original_image_width = im.shape[1]

            st.write('Select the dent region')
            
            # Select ROI, returns the rectangle coordinates
            rect = st_cropper(
                image,
                realtime_update=True,
                box_color='#0000FF',
                aspect_ratio=(1, 1),
                return_type="box"
            )

            left, top, width, height = tuple(map(int, rect.values()))

            manual_input = st.sidebar.checkbox("Take manual input from user", False)

            if manual_input:
                left = st.sidebar.text_input("Enter x_min")
                top = st.sidebar.text_input("Enter y_min")
                right = st.sidebar.text_input("Enter x_max")
                bottom = st.sidebar.text_input("Enter y_max")

                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)


                width = right - left
                height = bottom - top

            # st.write(f'y_min: {top}')
            # st.write(f'y_max: {top + height}')

            # st.write(f'x_min: {left}')
            # st.write(f'x_max: {left+width}')

            # Crop image according to ROI
            imCrop = im[int(top):int(top + height), int(left):int(left+width)]

            # displays the cropped image
            fig1, ax1 = plt.subplots()
            ax1.imshow(imCrop)
            st.pyplot(fig1)
            
            # resize the original image to match the depth image size
            img2 = cv2.resize(im,(192,256),interpolation = cv2.INTER_AREA)

            # scale the image coordinates to 256,192 to match depth coordinates
            scaled_left = scaled_width(left, original_image_width, 192)
            scaled_top = scaled_height(top, original_image_height, 256)
            scaled_right = scaled_width(left+width, original_image_width, 192)
            scaled_bottom = scaled_height(top + height, original_image_height, 256)

            
        except Exception as e:
            st.write("Error:", e)
        
    uploaded_file = st.sidebar.file_uploader("Choose a Depth file e.g. 'mirrorValue.txt'", type=["csv"])
    if uploaded_file is not None:
        # print(uploaded_file)
        # print(os.path.abspath(uploaded_file.name))
        try:
            df_car = pd.read_csv(uploaded_file)

            df_box = df_car.iloc[scaled_top:scaled_bottom,scaled_left:scaled_right]

            fig3, ax3 = plt.subplots()
            ax3.imshow(df_box)
            st.pyplot(fig3)

            ary_box = df_box.values
        except Exception as e:
            st.write("Error:", e)

    
    corner_choice = st.sidebar.radio(label="Corner Choice", options=["Top Left", "Bottom Left", "Bottom Right", "Top Right"])
    corner_dict = {
        "Top Left": 0,
        "Bottom Left": 1,
        "Bottom Right": 2,
        "Top Right": 3,
    }
    corner_value = corner_dict[corner_choice]

    st.write(f'Depth map using {corner_choice} corner')
    save_diff = st.sidebar.checkbox("Save diff array to disk", False)
    result = st.sidebar.button("Draw Depth Map")
    
    
    if result:

        f = open(f'{projectName}_Depth.txt','w')

        dent_diff, _, _ = extract_dent(ary_box,corner_value)
        dent_heatmap, _, _ = to_heatmap(dent_diff)
        upper_limit, lower_limit = get_limits(dent_diff)
        st.write(f'{corner_choice} Corner Depth Difference:  {(upper_limit-lower_limit)*1000} mm', )
        f.write(f'{corner_choice} Corner Depth Difference:  {(upper_limit-lower_limit)*1000} mm\n' )

        cum_avg_dent_diff = compute_cumm_avg_dent(ary_box)
        cum_avg_dent_heatmap, _, _ = to_heatmap(cum_avg_dent_diff)
        upper_limit, lower_limit = get_limits(cum_avg_dent_diff)
        st.write(f'Average Depth Difference: {(upper_limit-lower_limit)*1000} mm')
        f.write(f'Average Depth Difference:  {(upper_limit-lower_limit)*1000} mm\n')
        

        cum_min_dent_diff = compute_aggregate_min_dent(ary_box)
        cum_min_dent_heatmap, _, _ = to_heatmap(cum_min_dent_diff)
        upper_limit, lower_limit = get_limits(cum_min_dent_diff)
        st.write(f'Minimum Depth Difference: {(upper_limit-lower_limit)*1000} mm')
        f.write(f'Minimum Depth Difference:  {(upper_limit-lower_limit)*1000} mm')
        
        # Compute height and width of the dent   
        height, width = compute_Height_Width(df_box)

        st.write(f'Height: {height} mm, Width: {width} mm')
        f.write(f'Height: {height} mm, Width: {width} mm')
        f.close()

        if save_diff:
            np.savetxt(f'{projectName}_{corner_choice}_Corner_dent_diff.csv',dent_diff,delimiter = ',')
            np.savetxt(f'{projectName}_cum_min_dent_diff.csv',cum_min_dent_diff,delimiter = ',')
            np.savetxt(f'{projectName}_cum_avg_dent_diff.csv',cum_avg_dent_diff,delimiter = ',')
            

        fig4, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
        contour_plot = img2.copy()
        contour_plot[scaled_top:scaled_bottom,scaled_left:scaled_right] = dent_heatmap
        contour_plot2 = img2.copy()
        contour_plot2[scaled_top:scaled_bottom,scaled_left:scaled_right] = cum_avg_dent_heatmap
        contour_plot3 = img2.copy()
        contour_plot3[scaled_top:scaled_bottom,scaled_left:scaled_right] = cum_min_dent_heatmap
        ax4[0,0].imshow(contour_plot)
        ax4[0,0].set_title(f'{corner_choice} Corner' )
        ax4[0,1].imshow(contour_plot2)
        ax4[0,1].set_title('Avg')
        ax4[1,0].imshow(contour_plot3)
        ax4[1,0].set_title('Min')
        ax4[1,1].imshow(img2)
        ax4[1,1].set_title('Normal')
        fig4.tight_layout()
        st.pyplot(fig4)
        fig4.savefig(f'{projectName}_output.png', dpi=300)
    
if __name__ == "__main__":
    main()
"""
Author   : Serena Bonaretti 
Date     : Created on 29 March 2023.  Last update: xxx
License  : GPL GNU v3.0  
Email    : serena.bonaretti.research@gmail.com  
"""

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np


def show_slice_with_rods (slice_sitk, cx, cy, radii,  figure_size=(), title=""):

    """
    Visualize a slice with overlapped circles for the rods
    
    Parameters
    ----------
    slice_sitk: sitk image
        2D slice in SimpleITK format
    cx: list of floats
        x coordinate of the rod centers 
    cy: list of floats
        y coordinate of the rod centers
    radii: list of floats
        radii of the rods
    figure_size: tuple with two integers
        Size of the figure. Default is (4,4)
    title: string
        Title of the image
    """

   

    # convert slice from simpleitk to numpy
    one_slice_np = sitk.GetArrayFromImage(slice_sitk)

    # plot
    fig,ax = plt.subplots(1)
    
    # define figure size
    if figure_size == ():
        fig.set_figheight(4)
        fig.set_figwidth(4)
    else:
        fig.set_figheight(figure_size[0])
        fig.set_figwidth(figure_size[0])

        

    ax.imshow(one_slice_np, cmap=plt.cm.gray)

    # show the cicles
    colors = ["b", "r", "y", "c", "m", "g"]
    for xx,yy,radius,color in zip(cy, cx, radii, colors):
        circ = plt.Circle((yy,xx),radius, color=color, fill=False)
        ax.add_patch(circ)

    if title != "":
        ax.set_title(title)
    ax.axis("off");



def show_slice (img_slice, mask_slice=None, figure_size=(), title="", axis="off", rotate=0, flip=None, alpha=0.5):

    """
    Visualizes a slice using matplotlib
    
    Parameters
    ----------
    img_slice: np.array or sitk image
        2D slice in numpy or SimpleITK. The SimpleITK image will be converted to numpy for visualization with Matplotlib
    mask_slice_1: np.array or sitk image
        2D binary image in numpy or SimpleITK. The SimpleITK image will be converted to numpy for visualization with Matplotlib
    figure_size: tuple with two integers
        Size of the figure. Default is (4,4)
    title: string
        Title of the image
    axis: string
        Axis visibility. "on" or "off. Default is "off"
    rotate: int
        It rotates the slice counterclockwise. Possibilities are 90, 180, and 270. Default is 0
    flip: string
        Flips the image up or down, or left or right. Possible strings are "ud", "lr". Default is None 
    alpha: float
        Transparency of the mask. It must be a number between 0 (complete transparency) and 1 (no transparency). Defauls is 0.5

    """
    # check the inputs
    if img_slice.__class__ is not sitk.SimpleITK.Image and img_slice.__class__ is not np.ndarray:
        raise TypeError("img_slice has to be either a SimpleITK image or a NumPy array")
    if mask_slice != None:
        if mask_slice.__class__ is not sitk.SimpleITK.Image and img_slice.__class__ is not np.ndarray:
            raise TypeError("mask_slice has to be either a SimpleITK image or a NumPy array")
    if type(title) != str: 
        raise TypeError("title has to be a string")
    axis_options =["on", "off"]
    if axis not in axis_options:
        raise ValueError("Possible values for the axis parameter are: 'on', 'off'")
    rotate_options = [0, 90, 180, 270]
    if rotate not in rotate_options:
        raise ValueError("Possible values for the rotate parameter are: 0, 90, 180, 270. Default is 0")
    flip_options = ["ud", "lr", None]
    if flip not in flip_options:
        raise ValueError("Possible values for the flip parameter are: 'up', 'lr', and  None")
    if alpha <0 or alpha>1:
        raise ValueError("Alpha must be a number between 0 (complete transparency) and 1 (no transparency)") 

    # if the image is in SimpleITK format, convert it to numpy
    if img_slice.__class__ is sitk.SimpleITK.Image:
        img_slice = sitk.GetArrayFromImage(img_slice)
    # if the mask is in SimpleITK format, convert it to numpy
    if mask_slice.__class__ is sitk.SimpleITK.Image:
        mask_slice = sitk.GetArrayFromImage(mask_slice)

    # rotate the image (if requested)
    if rotate == 90:
        img_slice = np.rot90(img_slice)
        if mask_slice != None:
            mask_slice = np.rot90(mask_slice)
    elif rotate == 180:
        img_slice = np.rot90(img_slice, 2)
        if mask_slice != None:
            mask_slice = np.rot90(mask_slice,2)
    elif rotate == 270:
        img_slice = np.rot90(img_slice, 3)
        if mask_slice != None:
            mask_slice = np.rot90(mask_slice,3)

    # flip the image (if requested)
    if flip == "ud":
        img_slice = np.flipud(img_slice)
        if mask_slice != None:
            mask_slice = np.flipud(mask_slice)
    elif flip == "lr":
        img_slice = np.fliplr(img_slice)
        if mask_slice != None:
            mask_slice = np.fliplr(mask_slice)

    # define figure size
    if figure_size == ():
        plt.figure(figsize=(4,4))
    else:
        plt.figure(figsize=figure_size) 

    # show the image
    plt.imshow(img_slice, cmap=plt.cm.gray)
    # show the mask
    if mask_slice.__class__ is np.ndarray:
        plt.imshow(mask_slice, cmap='jet',  interpolation='none', alpha=alpha*(mask_slice>0))

    # beautify image
    if title != "":
        plt.title(title)

    if axis == "off":
        plt.axis('off');


def show_slice_per_plane(img, figure_size=()):
    
    """
    Visualizes one slice per plane using matplotlib
    
    Parameters
    ----------
    img: np.array or sitk image
        3D image in numpy or SimpleITK. The SimpleITK image will be converted to numpy for visualization with Matplotlib
    figure_size: tuple with two integers
        Size of the figure. Default is (4,8)
    
    """

    # check the inputs
    if img.__class__ is not sitk.SimpleITK.Image and img.__class__ is not np.ndarray:
        raise TypeError("img has to be either a SimpleITK image or a NumPy array")

    # get aspect ratio for viz that takes into account pixel size
    if img.__class__ is sitk.SimpleITK.Image:
        aspect_s = np.abs(img.GetSpacing()[2]/img.GetSpacing()[0])
        aspect_a = np.abs(img.GetSpacing()[1]/img.GetSpacing()[0])
        aspect_f = np.abs(img.GetSpacing()[2]/img.GetSpacing()[1])
    else:
        aspect_s = 1
        aspect_a = 1
        aspect_f = 1
    
    # if the image is in SimpleITK format, convert it to numpy
    if img.__class__ is sitk.SimpleITK.Image:
        img = sitk.GetArrayFromImage(img)

    # get slice ids (in Numpy)
    slice_id_a = img.shape[0]//2 
    slice_id_f = img.shape[1]//2 
    slice_id_s = img.shape[2]//2 

    # define figure size
    if figure_size == ():
        plt.figure(figsize=(20,0))
    else:
        plt.figure(figsize=figure_size) 

    # create figure
    plt.rcParams['figure.figsize'] = [figure_size[0], figure_size[1]] 
    fig     = plt.figure()                

    # show sagittal slice
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img[:,:,slice_id_s], 'gray', interpolation=None, aspect = aspect_s)
    ax1.set_title("Slice: " + str(slice_id_s))
    ax1.axis('off')

    # show axial slice ()
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(img[slice_id_a,:,:], 'gray', interpolation=None, aspect = aspect_a)
    ax2.set_title("Slice: " + str(slice_id_a))
    ax2.axis('off')

    # show frontal slice 
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(img[:,slice_id_f,:], 'gray', interpolation=None, aspect = aspect_f)
    ax3.set_title("Slice: " + str(slice_id_f))
    ax3.axis('off');
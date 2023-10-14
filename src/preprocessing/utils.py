import SimpleITK as sitk
import numpy as np
import os
import multiprocessing as mp

def read_dicom_series(folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def save_image(image, path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(image)
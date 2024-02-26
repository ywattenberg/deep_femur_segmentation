import SimpleITK as sitk
import itk

def read_aim_file(path, dtype="unsigned short"):
    """Read an AIM file and return a ITK image."""
    image_type = itk.Image[itk.ctype(dtype), 3]
    reader = itk.ImageFileReader[image_type].New()
    image_io = itk.ScancoImageIO.New()
    reader.SetImageIO(image_io)
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()
    # print(np.sum(itk.GetArrayFromImage(img)))
    return img
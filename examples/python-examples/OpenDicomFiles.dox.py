from meshlib import mrmeshpy

# Path to folder, that contains mutiple numbered files, like: 3DSlice1.dcm, 3DSlice2.dcm, 3DSlice3.dcm
# Folder can contain subfolders - each will be loaded separately
dicom_folder = 'path/to/folder'
# Load dicom objects from folder and it's subfolders
dicoms = mrmeshpy.VoxelsLoad.loadDicomsFolderTreeAsVdb(dicom_folder)
# Result contains list of multiple objects, each object is a separate dicom volume, that correspond to each folder
# Getting the first one here
dicom = dicoms[0]
# Check loading status
if dicom:
    # In case of success you can get mrmeshpy.VdbVolume object, that is common for meshlib library
    vdb_volume = dicom.value().vol
else:
    # print error if loading process failed
    print(dicom.error())

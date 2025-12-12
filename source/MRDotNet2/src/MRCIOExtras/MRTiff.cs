public static partial class MR
{
    public static partial class ImageLoad
    {
        /// loads from .tiff format
        /// Generated from function `MR::ImageLoad::fromTiff`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromTiff(ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromTiff", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromTiff(byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromTiff(__ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }
    }

    public static partial class ImageSave
    {
        /// saves in .tiff format
        /// Generated from function `MR::ImageSave::toTiff`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToTiff(MR.Const_Image image, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toTiff", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toTiff(MR.Const_Image._Underlying *image, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toTiff(image._UnderlyingPtr, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }
    }

    public static partial class DistanceMapSave
    {
        /// saves in .tiff format
        /// Generated from function `MR::DistanceMapSave::toTiff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToTiff(MR.Const_DistanceMap dmap, ReadOnlySpan<char> path, MR.Const_DistanceMapSaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSave_toTiff", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_DistanceMapSave_toTiff(MR.Const_DistanceMap._Underlying *dmap, byte *path, byte *path_end, MR.Const_DistanceMapSaveSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_DistanceMapSave_toTiff(dmap._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

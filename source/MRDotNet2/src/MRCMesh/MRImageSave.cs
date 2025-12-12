public static partial class MR
{
    public static partial class ImageSave
    {
        /// saves in .bmp format
        /// Generated from function `MR::ImageSave::toBmp`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToBmp(MR.Const_Image image, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toBmp", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toBmp(MR.Const_Image._Underlying *image, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toBmp(image._UnderlyingPtr, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// detects the format from file extension and save image to it  
        /// Generated from function `MR::ImageSave::toAnySupportedFormat`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Image image, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toAnySupportedFormat(MR.Const_Image._Underlying *image, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toAnySupportedFormat(image._UnderlyingPtr, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }
    }
}

public static partial class MR
{
    public static partial class ImageLoad
    {
        /// loads from .png format
        /// Generated from function `MR::ImageLoad::fromPng`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromPng(ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromPng_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromPng_std_filesystem_path(byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromPng_std_filesystem_path(__ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// Generated from function `MR::ImageLoad::fromPng`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromPng(MR.Std.Istream in_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromPng_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromPng_std_istream(MR.Std.Istream._Underlying *in_);
            return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromPng_std_istream(in_._UnderlyingPtr), is_owning: true));
        }
    }

    public static partial class ImageSave
    {
        /// saves in .png format
        /// Generated from function `MR::ImageSave::toPng`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPng(MR.Const_Image image, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toPng_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toPng_std_filesystem_path(MR.Const_Image._Underlying *image, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toPng_std_filesystem_path(image._UnderlyingPtr, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// Generated from function `MR::ImageSave::toPng`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPng(MR.Const_Image image, MR.Std.Ostream os)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toPng_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toPng_std_ostream(MR.Const_Image._Underlying *image, MR.Std.Ostream._Underlying *os);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toPng_std_ostream(image._UnderlyingPtr, os._UnderlyingPtr), is_owning: true));
        }
    }
}

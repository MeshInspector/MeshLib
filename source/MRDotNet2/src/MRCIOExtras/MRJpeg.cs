public static partial class MR
{
    public static partial class ImageLoad
    {
        /// loads from .jpg format
        /// Generated from function `MR::ImageLoad::fromJpeg`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromJpeg(ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromJpeg_1_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromJpeg_1_std_filesystem_path(byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromJpeg_1_std_filesystem_path(__ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// Generated from function `MR::ImageLoad::fromJpeg`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromJpeg(MR.Std.Istream in_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromJpeg_1_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromJpeg_1_std_istream(MR.Std.Istream._Underlying *in_);
            return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromJpeg_1_std_istream(in_._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::ImageLoad::fromJpeg`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromJpeg(byte? data, ulong size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromJpeg_2", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromJpeg_2(byte *data, ulong size);
            byte __deref_data = data.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromJpeg_2(data.HasValue ? &__deref_data : null, size), is_owning: true));
        }
    }

    public static partial class ImageSave
    {
        /// saves in .jpg format
        /// Generated from function `MR::ImageSave::toJpeg`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToJpeg(MR.Const_Image image, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageSave_toJpeg", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ImageSave_toJpeg(MR.Const_Image._Underlying *image, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ImageSave_toJpeg(image._UnderlyingPtr, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }
    }
}

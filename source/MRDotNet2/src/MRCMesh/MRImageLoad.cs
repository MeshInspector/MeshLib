public static partial class MR
{
    public static partial class ImageLoad
    {
        /// detects the format from file extension and loads image from it
        /// Generated from function `MR::ImageLoad::fromAnySupportedFormat`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRImage_StdString> FromAnySupportedFormat(ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImageLoad_fromAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_MRImage_StdString._Underlying *__MR_ImageLoad_fromAnySupportedFormat(byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRImage_StdString(__MR_ImageLoad_fromAnySupportedFormat(__ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }
    }
}

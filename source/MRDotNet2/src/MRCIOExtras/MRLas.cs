public static partial class MR
{
    public static partial class PointsLoad
    {
        /// loads from .las file
        /// Generated from function `MR::PointsLoad::fromLas`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromLas(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromLas_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromLas_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromLas_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromLas`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromLas(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromLas_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromLas_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromLas_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }
}

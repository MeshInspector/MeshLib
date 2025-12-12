public static partial class MR
{
    public static partial class PointsSave
    {
        /// save points without normals in textual .xyz file;
        /// each output line contains [x, y, z], where x, y, z are point coordinates
        /// Generated from function `MR::PointsSave::toXyz`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToXyz(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toXyz_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toXyz_std_filesystem_path(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toXyz_std_filesystem_path(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toXyz`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToXyz(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toXyz_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toXyz_std_ostream(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toXyz_std_ostream(points._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// save points with normals in textual .xyzn file;
        /// each output line contains [x, y, z, nx, ny, nz], where x, y, z are point coordinates and nx, ny, nz are the components of point normal
        /// Generated from function `MR::PointsSave::toXyzn`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToXyzn(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toXyzn_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toXyzn_std_filesystem_path(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toXyzn_std_filesystem_path(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toXyzn`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToXyzn(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toXyzn_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toXyzn_std_ostream(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toXyzn_std_ostream(points._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// save points with normals in .xyzn format, and save points without normals in .xyz format
        /// Generated from function `MR::PointsSave::toAsc`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAsc(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toAsc_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toAsc_std_filesystem_path(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toAsc_std_filesystem_path(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toAsc`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAsc(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toAsc_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toAsc_std_ostream(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toAsc_std_ostream(points._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .ply file
        /// Generated from function `MR::PointsSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toPly_std_filesystem_path(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toPly_std_filesystem_path(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toPly_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toPly_std_ostream(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toPly_std_ostream(points._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// detects the format from file extension and save points to it
        /// Generated from function `MR::PointsSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toAnySupportedFormat_3(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toAnySupportedFormat_3(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// extension in `*.ext` format
        /// Generated from function `MR::PointsSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_PointCloud points, ReadOnlySpan<char> extension, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toAnySupportedFormat_4", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toAnySupportedFormat_4(MR.Const_PointCloud._Underlying *points, byte *extension, byte *extension_end, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toAnySupportedFormat_4(points._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

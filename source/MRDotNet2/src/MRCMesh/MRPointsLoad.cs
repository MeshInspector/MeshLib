public static partial class MR
{
    public static partial class PointsLoad
    {
        /// loads from .csv, .asc, .xyz, .txt file
        /// Generated from function `MR::PointsLoad::fromText`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromText(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromText_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromText_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromText_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromText`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromText(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromText_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromText_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromText_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads from Laser scan plain data format (.pts) file
        /// Generated from function `MR::PointsLoad::fromPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromPts(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromPts_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromPts_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromPts_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromPts(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromPts_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromPts_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromPts_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads from .ply file
        /// Generated from function `MR::PointsLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromPly(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromPly_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromPly_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromPly(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromPly_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromPly_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromPly_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads from .obj file
        /// Generated from function `MR::PointsLoad::fromObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromObj(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromObj_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromObj_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromObj_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromObj(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromObj_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromObj_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromObj_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PointsLoad::fromDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromDxf(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromDxf_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromDxf_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromDxf_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromDxf(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromDxf_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromDxf_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromDxf_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// detects the format from file extension and loads points from it
        /// Generated from function `MR::PointsLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromAnySupportedFormat(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromAnySupportedFormat_2", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromAnySupportedFormat_2(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromAnySupportedFormat_2(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// extension in `*.ext` format
        /// Generated from function `MR::PointsLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromAnySupportedFormat(MR.Std.Istream in_, ReadOnlySpan<char> extension, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromAnySupportedFormat_3(MR.Std.Istream._Underlying *in_, byte *extension, byte *extension_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromAnySupportedFormat_3(in_._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

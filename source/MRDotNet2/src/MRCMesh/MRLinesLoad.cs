public static partial class MR
{
    public static partial class LinesLoad
    {
        /// loads polyline from file in internal MeshLib format
        /// Generated from function `MR::LinesLoad::fromMrLines`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromMrLines(ReadOnlySpan<char> file, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromMrLines_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromMrLines_std_filesystem_path(byte *file, byte *file_end, MR.Const_LinesLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromMrLines_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads polyline from stream in internal MeshLib format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::LinesLoad::fromMrLines`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromMrLines(MR.Std.Istream in_, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromMrLines_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromMrLines_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_LinesLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromMrLines_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads polyline from file in .PTS format
        /// Generated from function `MR::LinesLoad::fromPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromPts(ReadOnlySpan<char> file, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromPts_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromPts_std_filesystem_path(byte *file, byte *file_end, MR.Const_LinesLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromPts_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads polyline from stream in .PTS format
        /// Generated from function `MR::LinesLoad::fromPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromPts(MR.Std.Istream in_, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromPts_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromPts_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_LinesLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromPts_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads polyline from file in .PLY format
        /// Generated from function `MR::LinesLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromPly(ReadOnlySpan<char> file, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromPly_std_filesystem_path(byte *file, byte *file_end, MR.Const_LinesLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromPly_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads polyline from stream in .PLY format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::LinesLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromPly(MR.Std.Istream in_, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromPly_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromPly_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_LinesLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromPly_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads polyline from file in the format detected from file extension
        /// Generated from function `MR::LinesLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromAnySupportedFormat(ReadOnlySpan<char> file, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromAnySupportedFormat_2", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromAnySupportedFormat_2(byte *file, byte *file_end, MR.Const_LinesLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromAnySupportedFormat_2(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads polyline from stream in the format detected from given extension-string (`*.ext`);
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::LinesLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPolyline3_StdString> FromAnySupportedFormat(MR.Std.Istream in_, ReadOnlySpan<char> extension, MR.Const_LinesLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoad_fromAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_MRPolyline3_StdString._Underlying *__MR_LinesLoad_fromAnySupportedFormat_3(MR.Std.Istream._Underlying *in_, byte *extension, byte *extension_end, MR.Const_LinesLoadSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_MRPolyline3_StdString(__MR_LinesLoad_fromAnySupportedFormat_3(in_._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

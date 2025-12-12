public static partial class MR
{
    public static partial class LinesSave
    {
        /// saves in .mrlines file;
        /// SaveSettings::onlyValidPoints = true is ignored
        /// Generated from function `MR::LinesSave::toMrLines`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToMrLines(MR.Const_Polyline3 polyline, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toMrLines_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toMrLines_std_filesystem_path(MR.Const_Polyline3._Underlying *polyline, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toMrLines_std_filesystem_path(polyline._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::LinesSave::toMrLines`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToMrLines(MR.Const_Polyline3 polyline, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toMrLines_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toMrLines_std_ostream(MR.Const_Polyline3._Underlying *polyline, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toMrLines_std_ostream(polyline._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .pts file;
        /// SaveSettings::onlyValidPoints = false is ignored
        /// Generated from function `MR::LinesSave::toPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPts(MR.Const_Polyline3 polyline, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toPts_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toPts_std_filesystem_path(MR.Const_Polyline3._Underlying *polyline, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toPts_std_filesystem_path(polyline._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::LinesSave::toPts`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPts(MR.Const_Polyline3 polyline, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toPts_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toPts_std_ostream(MR.Const_Polyline3._Underlying *polyline, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toPts_std_ostream(polyline._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .dxf file;
        /// SaveSettings::onlyValidPoints = false is ignored
        /// Generated from function `MR::LinesSave::toDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToDxf(MR.Const_Polyline3 polyline, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toDxf_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toDxf_std_filesystem_path(MR.Const_Polyline3._Underlying *polyline, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toDxf_std_filesystem_path(polyline._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::LinesSave::toDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToDxf(MR.Const_Polyline3 polyline, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toDxf_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toDxf_std_ostream(MR.Const_Polyline3._Underlying *polyline, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toDxf_std_ostream(polyline._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .ply file
        /// Generated from function `MR::LinesSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_Polyline3 polyline, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toPly_std_filesystem_path(MR.Const_Polyline3._Underlying *polyline, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toPly_std_filesystem_path(polyline._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::LinesSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_Polyline3 polyline, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toPly_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toPly_std_ostream(MR.Const_Polyline3._Underlying *polyline, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toPly_std_ostream(polyline._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// detects the format from file extension and saves polyline in it
        /// Generated from function `MR::LinesSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Polyline3 polyline, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toAnySupportedFormat_3(MR.Const_Polyline3._Underlying *polyline, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toAnySupportedFormat_3(polyline._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// extension in `*.ext` format
        /// Generated from function `MR::LinesSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Polyline3 polyline, ReadOnlySpan<char> extension, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesSave_toAnySupportedFormat_4", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_LinesSave_toAnySupportedFormat_4(MR.Const_Polyline3._Underlying *polyline, byte *extension, byte *extension_end, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_LinesSave_toAnySupportedFormat_4(polyline._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

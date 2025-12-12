public static partial class MR
{
    public static partial class GcodeLoad
    {
        /// loads from *.gcode file (or any text file)
        /// Generated from function `MR::GcodeLoad::fromGcode`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdString_StdString> FromGcode(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeLoad_fromGcode_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdString_StdString._Underlying *__MR_GcodeLoad_fromGcode_std_filesystem_path(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorStdString_StdString(__MR_GcodeLoad_fromGcode_std_filesystem_path(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::GcodeLoad::fromGcode`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdString_StdString> FromGcode(MR.Std.Istream in_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeLoad_fromGcode_std_istream", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdString_StdString._Underlying *__MR_GcodeLoad_fromGcode_std_istream(MR.Std.Istream._Underlying *in_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_StdVectorStdString_StdString(__MR_GcodeLoad_fromGcode_std_istream(in_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// detects the format from file extension and loads mesh from it
        /// Generated from function `MR::GcodeLoad::fromAnySupportedFormat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdString_StdString> FromAnySupportedFormat(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeLoad_fromAnySupportedFormat_2", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdString_StdString._Underlying *__MR_GcodeLoad_fromAnySupportedFormat_2(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorStdString_StdString(__MR_GcodeLoad_fromAnySupportedFormat_2(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// extension in `*.ext` format
        /// Generated from function `MR::GcodeLoad::fromAnySupportedFormat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdString_StdString> FromAnySupportedFormat(MR.Std.Istream in_, ReadOnlySpan<char> extension, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeLoad_fromAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdString_StdString._Underlying *__MR_GcodeLoad_fromAnySupportedFormat_3(MR.Std.Istream._Underlying *in_, byte *extension, byte *extension_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorStdString_StdString(__MR_GcodeLoad_fromAnySupportedFormat_3(in_._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

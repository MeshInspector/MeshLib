public static partial class MR
{
    /// converts system encoded string to UTF8-encoded string
    /// Generated from function `MR::systemToUtf8`.
    public static unsafe MR.Misc._Moved<MR.Std.String> SystemToUtf8(ReadOnlySpan<char> system)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_systemToUtf8", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_systemToUtf8(byte *system, byte *system_end);
        byte[] __bytes_system = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(system.Length)];
        int __len_system = System.Text.Encoding.UTF8.GetBytes(system, __bytes_system);
        fixed (byte *__ptr_system = __bytes_system)
        {
            return MR.Misc.Move(new MR.Std.String(__MR_systemToUtf8(__ptr_system, __ptr_system + __len_system), is_owning: true));
        }
    }

    /// converts UTF8-encoded string to system encoded string,
    /// returns empty string if such conversion cannot be made
    /// Generated from function `MR::utf8ToSystem`.
    public static unsafe MR.Misc._Moved<MR.Std.String> Utf8ToSystem(ReadOnlySpan<char> utf8)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_utf8ToSystem", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_utf8ToSystem(byte *utf8, byte *utf8_end);
        byte[] __bytes_utf8 = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(utf8.Length)];
        int __len_utf8 = System.Text.Encoding.UTF8.GetBytes(utf8, __bytes_utf8);
        fixed (byte *__ptr_utf8 = __bytes_utf8)
        {
            return MR.Misc.Move(new MR.Std.String(__MR_utf8ToSystem(__ptr_utf8, __ptr_utf8 + __len_utf8), is_owning: true));
        }
    }

    /// Generated from function `MR::utf8string`.
    public static unsafe MR.Misc._Moved<MR.Std.String> Utf8string(ReadOnlySpan<char> path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_utf8string", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_utf8string(byte *path, byte *path_end);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Std.String(__MR_utf8string(__ptr_path, __ptr_path + __len_path), is_owning: true));
        }
    }

    /// given on input a valid utf8-encoded string, returns its substring starting at \p pos unicode symbol,
    /// and containing at most \p count unicode symbols (but res.size() can be more than \p count since a unicode symbol can be represented by more than 1 byte)
    /// Generated from function `MR::utf8substr`.
    public static unsafe MR.Misc._Moved<MR.Std.String> Utf8substr(byte? s, ulong pos, ulong count)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_utf8substr", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_utf8substr(byte *s, ulong pos, ulong count);
        byte __deref_s = s.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.String(__MR_utf8substr(s.HasValue ? &__deref_s : null, pos, count), is_owning: true));
    }

    /// converts given size in string:
    /// [0,1024) -> nnn bytes
    /// [1024,1024*1024) -> nnn.nn Kb
    /// [1024*1024,1024*1024*1024) -> nnn.nn Mb
    /// ...
    /// Generated from function `MR::bytesString`.
    public static unsafe MR.Misc._Moved<MR.Std.String> BytesString(ulong size)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bytesString", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_bytesString(ulong size);
        return MR.Misc.Move(new MR.Std.String(__MR_bytesString(size), is_owning: true));
    }

    /// returns true if the given character is any of prohibited in filenames in any of OSes
    /// https://stackoverflow.com/q/1976007/7325599
    /// Generated from function `MR::isProhibitedChar`.
    public static bool IsProhibitedChar(byte c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isProhibitedChar", ExactSpelling = true)]
        extern static byte __MR_isProhibitedChar(byte c);
        return __MR_isProhibitedChar(c) != 0;
    }

    /// returns true if line contains at least one character (c) for which isProhibitedChar(c)==true
    /// Generated from function `MR::hasProhibitedChars`.
    public static unsafe bool HasProhibitedChars(ReadOnlySpan<char> line)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasProhibitedChars", ExactSpelling = true)]
        extern static byte __MR_hasProhibitedChars(byte *line, byte *line_end);
        byte[] __bytes_line = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(line.Length)];
        int __len_line = System.Text.Encoding.UTF8.GetBytes(line, __bytes_line);
        fixed (byte *__ptr_line = __bytes_line)
        {
            return __MR_hasProhibitedChars(__ptr_line, __ptr_line + __len_line) != 0;
        }
    }

    /// replace all characters (c), where isProhibitedChar(c)==true, with `replacement` char
    /// Generated from function `MR::replaceProhibitedChars`.
    /// Parameter `replacement` defaults to `'_'`.
    public static unsafe MR.Misc._Moved<MR.Std.String> ReplaceProhibitedChars(ReadOnlySpan<char> line, byte? replacement = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_replaceProhibitedChars", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_replaceProhibitedChars(byte *line, byte *line_end, byte *replacement);
        byte[] __bytes_line = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(line.Length)];
        int __len_line = System.Text.Encoding.UTF8.GetBytes(line, __bytes_line);
        fixed (byte *__ptr_line = __bytes_line)
        {
            byte __deref_replacement = replacement.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.String(__MR_replaceProhibitedChars(__ptr_line, __ptr_line + __len_line, replacement.HasValue ? &__deref_replacement : null), is_owning: true));
        }
    }

    /// in case of empty vector, returns "Empty"
    /// in case of single input file.ext, returns ".EXT"
    /// in case of multiple files with same extension, returns ".EXTs"
    /// otherwise returns "Files"
    /// Generated from function `MR::commonFilesName`.
    public static unsafe MR.Misc._Moved<MR.Std.String> CommonFilesName(MR.Std.Const_Vector_StdFilesystemPath files)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_commonFilesName", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_commonFilesName(MR.Std.Const_Vector_StdFilesystemPath._Underlying *files);
        return MR.Misc.Move(new MR.Std.String(__MR_commonFilesName(files._UnderlyingPtr), is_owning: true));
    }

    /// returns given value rounded to given number of decimal digits
    /// Generated from function `MR::roundToPrecision`.
    public static double RoundToPrecision(double v, int precision)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_roundToPrecision_double", ExactSpelling = true)]
        extern static double __MR_roundToPrecision_double(double v, int precision);
        return __MR_roundToPrecision_double(v, precision);
    }

    /// returns given value rounded to given number of decimal digits
    /// Generated from function `MR::roundToPrecision`.
    public static float RoundToPrecision(float v, int precision)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_roundToPrecision_float", ExactSpelling = true)]
        extern static float __MR_roundToPrecision_float(float v, int precision);
        return __MR_roundToPrecision_float(v, precision);
    }

    // Returns message showed when loading is canceled
    /// Generated from function `MR::getCancelMessage`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetCancelMessage(ReadOnlySpan<char> path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getCancelMessage", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_getCancelMessage(byte *path, byte *path_end);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Std.String(__MR_getCancelMessage(__ptr_path, __ptr_path + __len_path), is_owning: true));
        }
    }

    /// return a copy of the string with all alphabetic ASCII characters replaced with upper-case variants
    /// Generated from function `MR::toLower`.
    public static unsafe MR.Misc._Moved<MR.Std.String> ToLower(ReadOnlySpan<char> str)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toLower", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_toLower(byte *str, byte *str_end);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            return MR.Misc.Move(new MR.Std.String(__MR_toLower(__ptr_str, __ptr_str + __len_str), is_owning: true));
        }
    }
}

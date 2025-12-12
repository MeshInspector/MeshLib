public static partial class MR
{
    /**
    * Finds the substring in the string.
    * \return position, npos if not found
    *
    */
    /// Generated from function `MR::findSubstringCaseInsensitive`.
    public static unsafe ulong FindSubstringCaseInsensitive(ReadOnlySpan<char> string_, ReadOnlySpan<char> substring)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSubstringCaseInsensitive", ExactSpelling = true)]
        extern static ulong __MR_findSubstringCaseInsensitive(byte *string_, byte *string__end, byte *substring, byte *substring_end);
        byte[] __bytes_string_ = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(string_.Length)];
        int __len_string_ = System.Text.Encoding.UTF8.GetBytes(string_, __bytes_string_);
        fixed (byte *__ptr_string_ = __bytes_string_)
        {
            byte[] __bytes_substring = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(substring.Length)];
            int __len_substring = System.Text.Encoding.UTF8.GetBytes(substring, __bytes_substring);
            fixed (byte *__ptr_substring = __bytes_substring)
            {
                return __MR_findSubstringCaseInsensitive(__ptr_string_, __ptr_string_ + __len_string_, __ptr_substring, __ptr_substring + __len_substring);
            }
        }
    }

    /**
    * Calculates Damerau-Levenshtein distance between to strings
    * \param outLeftRightAddition if provided return amount of insertions to the left and to the right
    *
    */
    /// Generated from function `MR::calcDamerauLevenshteinDistance`.
    /// Parameter `caseSensitive` defaults to `true`.
    public static unsafe int CalcDamerauLevenshteinDistance(ReadOnlySpan<char> stringA, ReadOnlySpan<char> stringB, bool? caseSensitive = null, MR.Misc.InOut<int>? outLeftRightAddition = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcDamerauLevenshteinDistance", ExactSpelling = true)]
        extern static int __MR_calcDamerauLevenshteinDistance(byte *stringA, byte *stringA_end, byte *stringB, byte *stringB_end, byte *caseSensitive, int *outLeftRightAddition);
        byte[] __bytes_stringA = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(stringA.Length)];
        int __len_stringA = System.Text.Encoding.UTF8.GetBytes(stringA, __bytes_stringA);
        fixed (byte *__ptr_stringA = __bytes_stringA)
        {
            byte[] __bytes_stringB = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(stringB.Length)];
            int __len_stringB = System.Text.Encoding.UTF8.GetBytes(stringB, __bytes_stringB);
            fixed (byte *__ptr_stringB = __bytes_stringB)
            {
                byte __deref_caseSensitive = caseSensitive.GetValueOrDefault() ? (byte)1 : (byte)0;
                int __value_outLeftRightAddition = outLeftRightAddition is not null ? outLeftRightAddition.Value : default(int);
                var __ret = __MR_calcDamerauLevenshteinDistance(__ptr_stringA, __ptr_stringA + __len_stringA, __ptr_stringB, __ptr_stringB + __len_stringB, caseSensitive.HasValue ? &__deref_caseSensitive : null, outLeftRightAddition is not null ? &__value_outLeftRightAddition : null);
                if (outLeftRightAddition is not null) outLeftRightAddition.Value = __value_outLeftRightAddition;
                return __ret;
            }
        }
    }

    /**
    * Splits given string by delimiter.
    * \return vector of split strings
    *
    */
    /// Generated from function `MR::split`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdString> Split(ReadOnlySpan<char> string_, ReadOnlySpan<char> delimiter)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_split", ExactSpelling = true)]
        extern static MR.Std.Vector_StdString._Underlying *__MR_split(byte *string_, byte *string__end, byte *delimiter, byte *delimiter_end);
        byte[] __bytes_string_ = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(string_.Length)];
        int __len_string_ = System.Text.Encoding.UTF8.GetBytes(string_, __bytes_string_);
        fixed (byte *__ptr_string_ = __bytes_string_)
        {
            byte[] __bytes_delimiter = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(delimiter.Length)];
            int __len_delimiter = System.Text.Encoding.UTF8.GetBytes(delimiter, __bytes_delimiter);
            fixed (byte *__ptr_delimiter = __bytes_delimiter)
            {
                return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_split(__ptr_string_, __ptr_string_ + __len_string_, __ptr_delimiter, __ptr_delimiter + __len_delimiter), is_owning: true));
            }
        }
    }

    /// Returns \param target with all \param from replaced with \param to, zero or more times.
    /// Generated from function `MR::replace`.
    public static unsafe MR.Misc._Moved<MR.Std.String> Replace(ReadOnlySpan<char> target, ReadOnlySpan<char> from, ReadOnlySpan<char> to)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_replace", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_replace(byte *target, byte *target_end, byte *from, byte *from_end, byte *to, byte *to_end);
        byte[] __bytes_target = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(target.Length)];
        int __len_target = System.Text.Encoding.UTF8.GetBytes(target, __bytes_target);
        fixed (byte *__ptr_target = __bytes_target)
        {
            byte[] __bytes_from = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(from.Length)];
            int __len_from = System.Text.Encoding.UTF8.GetBytes(from, __bytes_from);
            fixed (byte *__ptr_from = __bytes_from)
            {
                byte[] __bytes_to = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(to.Length)];
                int __len_to = System.Text.Encoding.UTF8.GetBytes(to, __bytes_to);
                fixed (byte *__ptr_to = __bytes_to)
                {
                    return MR.Misc.Move(new MR.Std.String(__MR_replace(__ptr_target, __ptr_target + __len_target, __ptr_from, __ptr_from + __len_from, __ptr_to, __ptr_to + __len_to), is_owning: true));
                }
            }
        }
    }

    /// Replaces \param from with \param to in \param target (in-place), zero or more times.
    /// Generated from function `MR::replaceInplace`.
    public static unsafe void ReplaceInplace(MR.Std.String target, ReadOnlySpan<char> from, ReadOnlySpan<char> to)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_replaceInplace", ExactSpelling = true)]
        extern static void __MR_replaceInplace(MR.Std.String._Underlying *target, byte *from, byte *from_end, byte *to, byte *to_end);
        byte[] __bytes_from = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(from.Length)];
        int __len_from = System.Text.Encoding.UTF8.GetBytes(from, __bytes_from);
        fixed (byte *__ptr_from = __bytes_from)
        {
            byte[] __bytes_to = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(to.Length)];
            int __len_to = System.Text.Encoding.UTF8.GetBytes(to, __bytes_to);
            fixed (byte *__ptr_to = __bytes_to)
            {
                __MR_replaceInplace(target._UnderlyingPtr, __ptr_from, __ptr_from + __len_from, __ptr_to, __ptr_to + __len_to);
            }
        }
    }

    /// Removes all whitespace character (detected by std::isspace) at the beginning and the end of string view
    /// Generated from function `MR::trim`.
    public static unsafe MR.Std.StringView Trim(ReadOnlySpan<char> str)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trim", ExactSpelling = true)]
        extern static MR.Std.StringView._Underlying *__MR_trim(byte *str, byte *str_end);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            return new(__MR_trim(__ptr_str, __ptr_str + __len_str), is_owning: true);
        }
    }

    /// Removes all whitespace character (detected by std::isspace) at the beginning of string view
    /// Generated from function `MR::trimLeft`.
    public static unsafe MR.Std.StringView TrimLeft(ReadOnlySpan<char> str)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trimLeft", ExactSpelling = true)]
        extern static MR.Std.StringView._Underlying *__MR_trimLeft(byte *str, byte *str_end);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            return new(__MR_trimLeft(__ptr_str, __ptr_str + __len_str), is_owning: true);
        }
    }

    /// Removes all whitespace character (detected by std::isspace) at the end of string view
    /// Generated from function `MR::trimRight`.
    public static unsafe MR.Std.StringView TrimRight(ReadOnlySpan<char> str)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trimRight", ExactSpelling = true)]
        extern static MR.Std.StringView._Underlying *__MR_trimRight(byte *str, byte *str_end);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            return new(__MR_trimRight(__ptr_str, __ptr_str + __len_str), is_owning: true);
        }
    }

    /// Returns true if `str` has at least one `{...}` formatting placeholder.
    /// Generated from function `MR::hasFormatPlaceholders`.
    public static unsafe bool HasFormatPlaceholders(ReadOnlySpan<char> str)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasFormatPlaceholders", ExactSpelling = true)]
        extern static byte __MR_hasFormatPlaceholders(byte *str, byte *str_end);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            return __MR_hasFormatPlaceholders(__ptr_str, __ptr_str + __len_str) != 0;
        }
    }
}

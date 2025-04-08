using System;
using System.Collections.Generic;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        /// use this function instead of Marshal.PtrToStringAnsi because:
        /// 1) C/C++ versions of MeshLib store all strings in UTF8 encoding even on Windows,
        /// 2) We cannot use Marshal.PtrToStringUTF8, which appears only in .Net Standard 2.1 not compatible with any .Net Framework
        /// https://stackoverflow.com/a/58358514/7325599
        public unsafe static string MarshalNativeUtf8ToManagedString(IntPtr pStringUtf8)
            => MarshalNativeUtf8ToManagedString((byte*)pStringUtf8);

        public unsafe static string MarshalNativeUtf8ToManagedString(byte* pStringUtf8)
        {
            var len = 0;
            while (pStringUtf8[len] != 0) len++;
            return Encoding.UTF8.GetString(pStringUtf8, len);
        }
    }
}

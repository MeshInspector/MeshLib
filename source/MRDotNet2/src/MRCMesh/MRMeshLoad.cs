public static partial class MR
{
    /// loads mesh from file in internal MeshLib format
    /// Generated from function `MR::loadMrmesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadMrmesh(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadMrmesh_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadMrmesh_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadMrmesh_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in internal MeshLib format;
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadMrmesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadMrmesh(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadMrmesh_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadMrmesh_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadMrmesh_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in .OFF format
    /// Generated from function `MR::loadOff`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadOff(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadOff_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadOff_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadOff_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in .OFF format
    /// Generated from function `MR::loadOff`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadOff(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadOff_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadOff_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadOff_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in .OBJ format
    /// Generated from function `MR::loadObj`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadObj(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadObj_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadObj_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadObj_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in .OBJ format;
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadObj`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadObj(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadObj_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadObj_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadObj_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in any .STL format: both binary and ASCII
    /// Generated from function `MR::loadStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadStl_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in any .STL format: both binary and ASCII;
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadStl_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in binary .STL format
    /// Generated from function `MR::loadBinaryStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadBinaryStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadBinaryStl_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadBinaryStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadBinaryStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in binary .STL format;
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadBinaryStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadBinaryStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadBinaryStl_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadBinaryStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadBinaryStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in textual .STL format
    /// Generated from function `MR::loadASCIIStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadASCIIStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadASCIIStl_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadASCIIStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadASCIIStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in textual .STL format
    /// Generated from function `MR::loadASCIIStl`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadASCIIStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadASCIIStl_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadASCIIStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadASCIIStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in .PLY format;
    /// Generated from function `MR::loadPly`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadPly(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadPly_const_std_filesystem_path_ref", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadPly_const_std_filesystem_path_ref(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadPly_const_std_filesystem_path_ref(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in .PLY format;
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadPly`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadPly(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadPly_std_istream_ref_MR_MeshLoadSettings", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadPly_std_istream_ref_MR_MeshLoadSettings(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadPly_std_istream_ref_MR_MeshLoadSettings(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in .DXF format;
    /// Generated from function `MR::loadDxf`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadDxf(ReadOnlySpan<char> path, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadDxf_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadDxf_std_filesystem_path(byte *path, byte *path_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadDxf_std_filesystem_path(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in .DXF format;
    /// Generated from function `MR::loadDxf`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadDxf(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadDxf_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadDxf_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadDxf_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// loads mesh from file in the format detected from file extension
    /// Generated from function `MR::loadMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadMesh(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadMesh_2", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadMesh_2(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadMesh_2(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads mesh from stream in the format detected from given extension-string (`*.ext`);
    /// important on Windows: in stream must be open in binary mode
    /// Generated from function `MR::loadMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> LoadMesh(MR.Std.Istream in_, ReadOnlySpan<char> extension, MR.Const_MeshLoadSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadMesh_3", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_loadMesh_3(MR.Std.Istream._Underlying *in_, byte *extension, byte *extension_end, MR.Const_MeshLoadSettings._Underlying *settings);
        byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
        int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
        fixed (byte *__ptr_extension = __bytes_extension)
        {
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_loadMesh_3(in_._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    public static partial class MeshLoad
    {
        /// loads mesh from file in internal MeshLib format
        /// Generated from function `MR::MeshLoad::fromMrmesh`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromMrmesh(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromMrmesh_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromMrmesh_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromMrmesh_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in internal MeshLib format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromMrmesh`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromMrmesh(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromMrmesh_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromMrmesh_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromMrmesh_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in .OFF format
        /// Generated from function `MR::MeshLoad::fromOff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromOff(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromOff_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromOff_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromOff_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in .OFF format
        /// Generated from function `MR::MeshLoad::fromOff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromOff(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromOff_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromOff_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromOff_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in .OBJ format
        /// Generated from function `MR::MeshLoad::fromObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromObj(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromObj_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromObj_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromObj_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in .OBJ format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromObj(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromObj_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromObj_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromObj_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in any .STL format: both binary and ASCII
        /// Generated from function `MR::MeshLoad::fromAnyStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromAnyStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromAnyStl_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromAnyStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromAnyStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in any .STL format: both binary and ASCII;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromAnyStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromAnyStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromAnyStl_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromAnyStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromAnyStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in binary .STL format
        /// Generated from function `MR::MeshLoad::fromBinaryStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromBinaryStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromBinaryStl_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromBinaryStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromBinaryStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in binary .STL format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromBinaryStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromBinaryStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromBinaryStl_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromBinaryStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromBinaryStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in textual .STL format
        /// Generated from function `MR::MeshLoad::fromASCIIStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromASCIIStl(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromASCIIStl_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromASCIIStl_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromASCIIStl_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in textual .STL format
        /// Generated from function `MR::MeshLoad::fromASCIIStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromASCIIStl(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromASCIIStl_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromASCIIStl_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromASCIIStl_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in .PLY format;
        /// Generated from function `MR::MeshLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromPly(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromPly_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromPly_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in .PLY format;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromPly(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromPly_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromPly_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromPly_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in .DXF format;
        /// Generated from function `MR::MeshLoad::fromDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromDxf(ReadOnlySpan<char> path, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromDxf_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromDxf_std_filesystem_path(byte *path, byte *path_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromDxf_std_filesystem_path(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in .DXF format;
        /// Generated from function `MR::MeshLoad::fromDxf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromDxf(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromDxf_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromDxf_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromDxf_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// loads mesh from file in the format detected from file extension
        /// Generated from function `MR::MeshLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromAnySupportedFormat(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromAnySupportedFormat_2", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromAnySupportedFormat_2(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromAnySupportedFormat_2(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads mesh from stream in the format detected from given extension-string (`*.ext`);
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::MeshLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromAnySupportedFormat(MR.Std.Istream in_, ReadOnlySpan<char> extension, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromAnySupportedFormat_3(MR.Std.Istream._Underlying *in_, byte *extension, byte *extension_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromAnySupportedFormat_3(in_._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

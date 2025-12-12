public static partial class MR
{
    public static partial class MeshSave
    {
        /// saves in internal file format;
        /// SaveSettings::onlyValidPoints = true is ignored
        /// Generated from function `MR::MeshSave::toMrmesh`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToMrmesh(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toMrmesh_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toMrmesh_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toMrmesh_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toMrmesh`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToMrmesh(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toMrmesh_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toMrmesh_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toMrmesh_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .off file
        /// Generated from function `MR::MeshSave::toOff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToOff(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toOff_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toOff_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toOff_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toOff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToOff(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toOff_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toOff_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toOff_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .obj file
        /// \param firstVertId is the index of first mesh vertex in the output file (if this object is not the first there)
        /// Generated from function `MR::MeshSave::toObj`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToObj(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings settings, int firstVertId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toObj_4_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toObj_4_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings, int firstVertId);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toObj_4_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings._UnderlyingPtr, firstVertId), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toObj`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToObj(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings settings, int firstVertId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toObj_4_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toObj_4_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings, int firstVertId);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toObj_4_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings._UnderlyingPtr, firstVertId), is_owning: true));
        }

        /// Generated from function `MR::MeshSave::toObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToObj(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toObj_3_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toObj_3_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toObj_3_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toObj`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToObj(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toObj_3_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toObj_3_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toObj_3_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in binary .stl file;
        /// SaveSettings::onlyValidPoints = false is ignored
        /// Generated from function `MR::MeshSave::toBinaryStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToBinaryStl(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toBinaryStl_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toBinaryStl_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toBinaryStl_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toBinaryStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToBinaryStl(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toBinaryStl_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toBinaryStl_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toBinaryStl_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in textual .stl file;
        /// SaveSettings::onlyValidPoints = false is ignored
        /// Generated from function `MR::MeshSave::toAsciiStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAsciiStl(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toAsciiStl_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toAsciiStl_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toAsciiStl_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toAsciiStl`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAsciiStl(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toAsciiStl_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toAsciiStl_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toAsciiStl_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .ply file
        /// Generated from function `MR::MeshSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toPly_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toPly_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toPly_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toPly`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToPly(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toPly_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toPly_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toPly_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in 3mf .model file
        /// Generated from function `MR::MeshSave::toModel3mf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToModel3mf(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toModel3mf_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toModel3mf_std_filesystem_path(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toModel3mf_std_filesystem_path(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toModel3mf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToModel3mf(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toModel3mf_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toModel3mf_std_ostream(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toModel3mf_std_ostream(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// saves in .3mf file
        /// Generated from function `MR::MeshSave::to3mf`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> To3mf(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_to3mf", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_to3mf(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_to3mf(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// detects the format from file extension and save mesh to it
        /// Generated from function `MR::MeshSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toAnySupportedFormat_3", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toAnySupportedFormat_3(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toAnySupportedFormat_3(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// extension in `*.ext` format
        /// Generated from function `MR::MeshSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Mesh mesh, ReadOnlySpan<char> extension, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toAnySupportedFormat_4", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toAnySupportedFormat_4(MR.Const_Mesh._Underlying *mesh, byte *extension, byte *extension_end, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
            int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
            fixed (byte *__ptr_extension = __bytes_extension)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toAnySupportedFormat_4(mesh._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

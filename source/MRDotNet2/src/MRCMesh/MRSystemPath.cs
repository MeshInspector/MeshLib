public static partial class MR
{
    /// system directory path manager
    /// Generated from class `MR::SystemPath`.
    /// This is the const half of the class.
    public class Const_SystemPath : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SystemPath(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_Destroy", ExactSpelling = true)]
            extern static void __MR_SystemPath_Destroy(_Underlying *_this);
            __MR_SystemPath_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SystemPath() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SystemPath() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SystemPath._Underlying *__MR_SystemPath_DefaultConstruct();
            _UnderlyingPtr = __MR_SystemPath_DefaultConstruct();
        }

        /// Generated from constructor `MR::SystemPath::SystemPath`.
        public unsafe Const_SystemPath(MR.Const_SystemPath _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SystemPath._Underlying *__MR_SystemPath_ConstructFromAnother(MR.SystemPath._Underlying *_other);
            _UnderlyingPtr = __MR_SystemPath_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// get the directory path for specified category
        /// Generated from method `MR::SystemPath::getDirectory`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetDirectory(MR.SystemPath.Directory dir)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getDirectory", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_SystemPath_getDirectory(MR.SystemPath.Directory dir);
            return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_SystemPath_getDirectory(dir), is_owning: true));
        }

        /// override the directory path for specified category, useful for custom configurations
        /// Generated from method `MR::SystemPath::overrideDirectory`.
        public static unsafe void OverrideDirectory(MR.SystemPath.Directory dir, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_overrideDirectory", ExactSpelling = true)]
            extern static void __MR_SystemPath_overrideDirectory(MR.SystemPath.Directory dir, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                __MR_SystemPath_overrideDirectory(dir, __ptr_path, __ptr_path + __len_path);
            }
        }

        /// get the resource files' directory path
        /// Generated from method `MR::SystemPath::getResourcesDirectory`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetResourcesDirectory()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getResourcesDirectory", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_SystemPath_getResourcesDirectory();
            return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_SystemPath_getResourcesDirectory(), is_owning: true));
        }

        /// get the font files' directory path
        /// Generated from method `MR::SystemPath::getFontsDirectory`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetFontsDirectory()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getFontsDirectory", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_SystemPath_getFontsDirectory();
            return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_SystemPath_getFontsDirectory(), is_owning: true));
        }

        /// get the plugin binaries' directory path
        /// Generated from method `MR::SystemPath::getPluginsDirectory`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetPluginsDirectory()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getPluginsDirectory", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_SystemPath_getPluginsDirectory();
            return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_SystemPath_getPluginsDirectory(), is_owning: true));
        }

        /// get the Python modules' binaries' directory path
        /// Generated from method `MR::SystemPath::getPythonModulesDirectory`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetPythonModulesDirectory()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getPythonModulesDirectory", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_SystemPath_getPythonModulesDirectory();
            return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_SystemPath_getPythonModulesDirectory(), is_owning: true));
        }

        /// get name all system fonts that have italics, bold, bold italics
        /// Generated from method `MR::SystemPath::getSystemFonts`.
        public static unsafe MR.Std.Const_Vector_StdArrayStdFilesystemPath4 GetSystemFonts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_getSystemFonts", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdArrayStdFilesystemPath4._Underlying *__MR_SystemPath_getSystemFonts();
            return new(__MR_SystemPath_getSystemFonts(), is_owning: false);
        }

        /// directory category
        public enum Directory : int
        {
            /// resources (.json, .png)
            Resources = 0,
            /// fonts (.ttf, .otf)
            Fonts = 1,
            /// plugins (.dll, .so, .dylib)
            Plugins = 2,
            /// Python modules (.pyd, .so, .dylib, .pyi)
            PythonModules = 3,
            /// Python modules (.pyd, .so, .dylib, .pyi)
            Count = 4,
        }

        // supported types of system fonts fonts
        public enum SystemFontType : int
        {
            Regular = 0,
            Bold = 1,
            Italic = 2,
            BoldItalic = 3,
            Count = 4,
        }
    }

    /// system directory path manager
    /// Generated from class `MR::SystemPath`.
    /// This is the non-const half of the class.
    public class SystemPath : Const_SystemPath
    {
        internal unsafe SystemPath(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe SystemPath() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SystemPath._Underlying *__MR_SystemPath_DefaultConstruct();
            _UnderlyingPtr = __MR_SystemPath_DefaultConstruct();
        }

        /// Generated from constructor `MR::SystemPath::SystemPath`.
        public unsafe SystemPath(MR.Const_SystemPath _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SystemPath._Underlying *__MR_SystemPath_ConstructFromAnother(MR.SystemPath._Underlying *_other);
            _UnderlyingPtr = __MR_SystemPath_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SystemPath::operator=`.
        public unsafe MR.SystemPath Assign(MR.Const_SystemPath _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemPath_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SystemPath._Underlying *__MR_SystemPath_AssignFromAnother(_Underlying *_this, MR.SystemPath._Underlying *_other);
            return new(__MR_SystemPath_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SystemPath` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SystemPath`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SystemPath`/`Const_SystemPath` directly.
    public class _InOptMut_SystemPath
    {
        public SystemPath? Opt;

        public _InOptMut_SystemPath() {}
        public _InOptMut_SystemPath(SystemPath value) {Opt = value;}
        public static implicit operator _InOptMut_SystemPath(SystemPath value) {return new(value);}
    }

    /// This is used for optional parameters of class `SystemPath` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SystemPath`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SystemPath`/`Const_SystemPath` to pass it to the function.
    public class _InOptConst_SystemPath
    {
        public Const_SystemPath? Opt;

        public _InOptConst_SystemPath() {}
        public _InOptConst_SystemPath(Const_SystemPath value) {Opt = value;}
        public static implicit operator _InOptConst_SystemPath(Const_SystemPath value) {return new(value);}
    }
}

public static partial class MR
{
    /// Generated from class `MR::SystemMemory`.
    /// This is the const half of the class.
    public class Const_SystemMemory : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SystemMemory(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_Destroy", ExactSpelling = true)]
            extern static void __MR_SystemMemory_Destroy(_Underlying *_this);
            __MR_SystemMemory_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SystemMemory() {Dispose(false);}

        /// total amount of physical memory in the system, in bytes (0 if no info)
        public unsafe ulong PhysicalTotal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_Get_physicalTotal", ExactSpelling = true)]
                extern static ulong *__MR_SystemMemory_Get_physicalTotal(_Underlying *_this);
                return *__MR_SystemMemory_Get_physicalTotal(_UnderlyingPtr);
            }
        }

        /// available amount of physical memory in the system, in bytes (0 if no info)
        public unsafe ulong PhysicalAvailable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_Get_physicalAvailable", ExactSpelling = true)]
                extern static ulong *__MR_SystemMemory_Get_physicalAvailable(_Underlying *_this);
                return *__MR_SystemMemory_Get_physicalAvailable(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SystemMemory() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_DefaultConstruct();
            _UnderlyingPtr = __MR_SystemMemory_DefaultConstruct();
        }

        /// Constructs `MR::SystemMemory` elementwise.
        public unsafe Const_SystemMemory(ulong physicalTotal, ulong physicalAvailable) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_ConstructFrom", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_ConstructFrom(ulong physicalTotal, ulong physicalAvailable);
            _UnderlyingPtr = __MR_SystemMemory_ConstructFrom(physicalTotal, physicalAvailable);
        }

        /// Generated from constructor `MR::SystemMemory::SystemMemory`.
        public unsafe Const_SystemMemory(MR.Const_SystemMemory _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_ConstructFromAnother(MR.SystemMemory._Underlying *_other);
            _UnderlyingPtr = __MR_SystemMemory_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::SystemMemory`.
    /// This is the non-const half of the class.
    public class SystemMemory : Const_SystemMemory
    {
        internal unsafe SystemMemory(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// total amount of physical memory in the system, in bytes (0 if no info)
        public new unsafe ref ulong PhysicalTotal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_GetMutable_physicalTotal", ExactSpelling = true)]
                extern static ulong *__MR_SystemMemory_GetMutable_physicalTotal(_Underlying *_this);
                return ref *__MR_SystemMemory_GetMutable_physicalTotal(_UnderlyingPtr);
            }
        }

        /// available amount of physical memory in the system, in bytes (0 if no info)
        public new unsafe ref ulong PhysicalAvailable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_GetMutable_physicalAvailable", ExactSpelling = true)]
                extern static ulong *__MR_SystemMemory_GetMutable_physicalAvailable(_Underlying *_this);
                return ref *__MR_SystemMemory_GetMutable_physicalAvailable(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SystemMemory() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_DefaultConstruct();
            _UnderlyingPtr = __MR_SystemMemory_DefaultConstruct();
        }

        /// Constructs `MR::SystemMemory` elementwise.
        public unsafe SystemMemory(ulong physicalTotal, ulong physicalAvailable) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_ConstructFrom", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_ConstructFrom(ulong physicalTotal, ulong physicalAvailable);
            _UnderlyingPtr = __MR_SystemMemory_ConstructFrom(physicalTotal, physicalAvailable);
        }

        /// Generated from constructor `MR::SystemMemory::SystemMemory`.
        public unsafe SystemMemory(MR.Const_SystemMemory _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_ConstructFromAnother(MR.SystemMemory._Underlying *_other);
            _UnderlyingPtr = __MR_SystemMemory_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SystemMemory::operator=`.
        public unsafe MR.SystemMemory Assign(MR.Const_SystemMemory _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SystemMemory_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SystemMemory._Underlying *__MR_SystemMemory_AssignFromAnother(_Underlying *_this, MR.SystemMemory._Underlying *_other);
            return new(__MR_SystemMemory_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SystemMemory` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SystemMemory`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SystemMemory`/`Const_SystemMemory` directly.
    public class _InOptMut_SystemMemory
    {
        public SystemMemory? Opt;

        public _InOptMut_SystemMemory() {}
        public _InOptMut_SystemMemory(SystemMemory value) {Opt = value;}
        public static implicit operator _InOptMut_SystemMemory(SystemMemory value) {return new(value);}
    }

    /// This is used for optional parameters of class `SystemMemory` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SystemMemory`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SystemMemory`/`Const_SystemMemory` to pass it to the function.
    public class _InOptConst_SystemMemory
    {
        public Const_SystemMemory? Opt;

        public _InOptConst_SystemMemory() {}
        public _InOptConst_SystemMemory(Const_SystemMemory value) {Opt = value;}
        public static implicit operator _InOptConst_SystemMemory(Const_SystemMemory value) {return new(value);}
    }

    // sets debug name for the current thread
    /// Generated from function `MR::SetCurrentThreadName`.
    public static unsafe void SetCurrentThreadName(byte? name)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetCurrentThreadName", ExactSpelling = true)]
        extern static void __MR_SetCurrentThreadName(byte *name);
        byte __deref_name = name.GetValueOrDefault();
        __MR_SetCurrentThreadName(name.HasValue ? &__deref_name : null);
    }

    // return path to the folder with user config file(s)
    /// Generated from function `MR::getUserConfigDir`.
    public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetUserConfigDir()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getUserConfigDir", ExactSpelling = true)]
        extern static MR.Std.Filesystem.Path._Underlying *__MR_getUserConfigDir();
        return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_getUserConfigDir(), is_owning: true));
    }

    // returns path of config file in APPDATA
    /// Generated from function `MR::getUserConfigFilePath`.
    public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetUserConfigFilePath()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getUserConfigFilePath", ExactSpelling = true)]
        extern static MR.Std.Filesystem.Path._Underlying *__MR_getUserConfigFilePath();
        return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_getUserConfigFilePath(), is_owning: true));
    }

    // returns temp directory
    /// Generated from function `MR::GetTempDirectory`.
    public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetTempDirectory()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetTempDirectory", ExactSpelling = true)]
        extern static MR.Std.Filesystem.Path._Underlying *__MR_GetTempDirectory();
        return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_GetTempDirectory(), is_owning: true));
    }

    // returns home directory
    /// Generated from function `MR::GetHomeDirectory`.
    public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> GetHomeDirectory()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetHomeDirectory", ExactSpelling = true)]
        extern static MR.Std.Filesystem.Path._Underlying *__MR_GetHomeDirectory();
        return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_GetHomeDirectory(), is_owning: true));
    }

    // returns version of MR
    /// Generated from function `MR::GetMRVersionString`.
    public static unsafe byte? GetMRVersionString()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetMRVersionString", ExactSpelling = true)]
        extern static byte *__MR_GetMRVersionString();
        var __ret = __MR_GetMRVersionString();
        return __ret is not null ? *__ret : null;
    }

    // Opens given link in default browser
    /// Generated from function `MR::OpenLink`.
    public static unsafe void OpenLink(ReadOnlySpan<char> url)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenLink", ExactSpelling = true)]
        extern static void __MR_OpenLink(byte *url, byte *url_end);
        byte[] __bytes_url = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(url.Length)];
        int __len_url = System.Text.Encoding.UTF8.GetBytes(url, __bytes_url);
        fixed (byte *__ptr_url = __bytes_url)
        {
            __MR_OpenLink(__ptr_url, __ptr_url + __len_url);
        }
    }

    // Opens given file (or directory) in associated application
    /// Generated from function `MR::OpenDocument`.
    public static unsafe bool OpenDocument(ReadOnlySpan<char> path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenDocument", ExactSpelling = true)]
        extern static byte __MR_OpenDocument(byte *path, byte *path_end);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return __MR_OpenDocument(__ptr_path, __ptr_path + __len_path) != 0;
        }
    }

    // returns string identification of the CPU
    /// Generated from function `MR::GetCpuId`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetCpuId()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetCpuId", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_GetCpuId();
        return MR.Misc.Move(new MR.Std.String(__MR_GetCpuId(), is_owning: true));
    }

    // returns string with OS name with details
    /// Generated from function `MR::GetDetailedOSName`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetDetailedOSName()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetDetailedOSName", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_GetDetailedOSName();
        return MR.Misc.Move(new MR.Std.String(__MR_GetDetailedOSName(), is_owning: true));
    }

    // returns string identification of the OS
    /// Generated from function `MR::getOSNoSpaces`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetOSNoSpaces()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getOSNoSpaces", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_getOSNoSpaces();
        return MR.Misc.Move(new MR.Std.String(__MR_getOSNoSpaces(), is_owning: true));
    }

    // sets new handler for operator new if needed for some platforms
    /// Generated from function `MR::setNewHandlerIfNeeded`.
    public static void SetNewHandlerIfNeeded()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setNewHandlerIfNeeded", ExactSpelling = true)]
        extern static void __MR_setNewHandlerIfNeeded();
        __MR_setNewHandlerIfNeeded();
    }

    /// returns string representation of the current stacktrace
    /// Generated from function `MR::getCurrentStacktrace`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetCurrentStacktrace()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getCurrentStacktrace", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_getCurrentStacktrace();
        return MR.Misc.Move(new MR.Std.String(__MR_getCurrentStacktrace(), is_owning: true));
    }

    /// return information about memory available in the system
    /// Generated from function `MR::getSystemMemory`.
    public static unsafe MR.SystemMemory GetSystemMemory()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getSystemMemory", ExactSpelling = true)]
        extern static MR.SystemMemory._Underlying *__MR_getSystemMemory();
        return new(__MR_getSystemMemory(), is_owning: true);
    }

    /// Setups logger:
    /// 1) makes stdout sink
    /// 2) makes file sink (MRLog.txt)
    /// 3) redirect std streams to logger
    /// 4) print stacktrace on crash (not in wasm)
    /// log level - trace
    /// Generated from function `MR::setupLoggerByDefault`.
    public static void SetupLoggerByDefault()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setupLoggerByDefault", ExactSpelling = true)]
        extern static void __MR_setupLoggerByDefault();
        __MR_setupLoggerByDefault();
    }
}

public static partial class MR
{
    /// slice information
    /// \sa SliceInfo
    /// Generated from class `MR::SliceInfoBase`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SliceInfo`
    /// This is the const half of the class.
    public class Const_SliceInfoBase : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SliceInfoBase(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_Destroy", ExactSpelling = true)]
            extern static void __MR_SliceInfoBase_Destroy(_Underlying *_this);
            __MR_SliceInfoBase_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SliceInfoBase() {Dispose(false);}

        /// instance number
        public unsafe int InstanceNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_Get_instanceNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfoBase_Get_instanceNum(_Underlying *_this);
                return *__MR_SliceInfoBase_Get_instanceNum(_UnderlyingPtr);
            }
        }

        /// layer height
        public unsafe double Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_Get_z", ExactSpelling = true)]
                extern static double *__MR_SliceInfoBase_Get_z(_Underlying *_this);
                return *__MR_SliceInfoBase_Get_z(_UnderlyingPtr);
            }
        }

        /// file index
        public unsafe int FileNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_Get_fileNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfoBase_Get_fileNum(_Underlying *_this);
                return *__MR_SliceInfoBase_Get_fileNum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SliceInfoBase() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_DefaultConstruct();
            _UnderlyingPtr = __MR_SliceInfoBase_DefaultConstruct();
        }

        /// Constructs `MR::SliceInfoBase` elementwise.
        public unsafe Const_SliceInfoBase(int instanceNum, double z, int fileNum) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_ConstructFrom", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_ConstructFrom(int instanceNum, double z, int fileNum);
            _UnderlyingPtr = __MR_SliceInfoBase_ConstructFrom(instanceNum, z, fileNum);
        }

        /// Generated from constructor `MR::SliceInfoBase::SliceInfoBase`.
        public unsafe Const_SliceInfoBase(MR.Const_SliceInfoBase _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_ConstructFromAnother(MR.SliceInfoBase._Underlying *_other);
            _UnderlyingPtr = __MR_SliceInfoBase_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// slice information
    /// \sa SliceInfo
    /// Generated from class `MR::SliceInfoBase`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SliceInfo`
    /// This is the non-const half of the class.
    public class SliceInfoBase : Const_SliceInfoBase
    {
        internal unsafe SliceInfoBase(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// instance number
        public new unsafe ref int InstanceNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_GetMutable_instanceNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfoBase_GetMutable_instanceNum(_Underlying *_this);
                return ref *__MR_SliceInfoBase_GetMutable_instanceNum(_UnderlyingPtr);
            }
        }

        /// layer height
        public new unsafe ref double Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_GetMutable_z", ExactSpelling = true)]
                extern static double *__MR_SliceInfoBase_GetMutable_z(_Underlying *_this);
                return ref *__MR_SliceInfoBase_GetMutable_z(_UnderlyingPtr);
            }
        }

        /// file index
        public new unsafe ref int FileNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_GetMutable_fileNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfoBase_GetMutable_fileNum(_Underlying *_this);
                return ref *__MR_SliceInfoBase_GetMutable_fileNum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SliceInfoBase() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_DefaultConstruct();
            _UnderlyingPtr = __MR_SliceInfoBase_DefaultConstruct();
        }

        /// Constructs `MR::SliceInfoBase` elementwise.
        public unsafe SliceInfoBase(int instanceNum, double z, int fileNum) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_ConstructFrom", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_ConstructFrom(int instanceNum, double z, int fileNum);
            _UnderlyingPtr = __MR_SliceInfoBase_ConstructFrom(instanceNum, z, fileNum);
        }

        /// Generated from constructor `MR::SliceInfoBase::SliceInfoBase`.
        public unsafe SliceInfoBase(MR.Const_SliceInfoBase _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_ConstructFromAnother(MR.SliceInfoBase._Underlying *_other);
            _UnderlyingPtr = __MR_SliceInfoBase_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SliceInfoBase::operator=`.
        public unsafe MR.SliceInfoBase Assign(MR.Const_SliceInfoBase _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfoBase_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfoBase_AssignFromAnother(_Underlying *_this, MR.SliceInfoBase._Underlying *_other);
            return new(__MR_SliceInfoBase_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SliceInfoBase` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SliceInfoBase`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SliceInfoBase`/`Const_SliceInfoBase` directly.
    public class _InOptMut_SliceInfoBase
    {
        public SliceInfoBase? Opt;

        public _InOptMut_SliceInfoBase() {}
        public _InOptMut_SliceInfoBase(SliceInfoBase value) {Opt = value;}
        public static implicit operator _InOptMut_SliceInfoBase(SliceInfoBase value) {return new(value);}
    }

    /// This is used for optional parameters of class `SliceInfoBase` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SliceInfoBase`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SliceInfoBase`/`Const_SliceInfoBase` to pass it to the function.
    public class _InOptConst_SliceInfoBase
    {
        public Const_SliceInfoBase? Opt;

        public _InOptConst_SliceInfoBase() {}
        public _InOptConst_SliceInfoBase(Const_SliceInfoBase value) {Opt = value;}
        public static implicit operator _InOptConst_SliceInfoBase(Const_SliceInfoBase value) {return new(value);}
    }

    /// slice information
    /// these fields will be ignored in sorting
    /// \sa SliceInfoBase
    /// Generated from class `MR::SliceInfo`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SliceInfoBase`
    /// This is the const half of the class.
    public class Const_SliceInfo : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SliceInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_Destroy", ExactSpelling = true)]
            extern static void __MR_SliceInfo_Destroy(_Underlying *_this);
            __MR_SliceInfo_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SliceInfo() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_SliceInfoBase(Const_SliceInfo self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_UpcastTo_MR_SliceInfoBase", ExactSpelling = true)]
            extern static MR.Const_SliceInfoBase._Underlying *__MR_SliceInfo_UpcastTo_MR_SliceInfoBase(_Underlying *_this);
            MR.Const_SliceInfoBase ret = new(__MR_SliceInfo_UpcastTo_MR_SliceInfoBase(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// image position
        public unsafe MR.Const_Vector3d ImagePos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_Get_imagePos", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_SliceInfo_Get_imagePos(_Underlying *_this);
                return new(__MR_SliceInfo_Get_imagePos(_UnderlyingPtr), is_owning: false);
            }
        }

        /// instance number
        public unsafe int InstanceNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_Get_instanceNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfo_Get_instanceNum(_Underlying *_this);
                return *__MR_SliceInfo_Get_instanceNum(_UnderlyingPtr);
            }
        }

        /// layer height
        public unsafe double Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_Get_z", ExactSpelling = true)]
                extern static double *__MR_SliceInfo_Get_z(_Underlying *_this);
                return *__MR_SliceInfo_Get_z(_UnderlyingPtr);
            }
        }

        /// file index
        public unsafe int FileNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_Get_fileNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfo_Get_fileNum(_Underlying *_this);
                return *__MR_SliceInfo_Get_fileNum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SliceInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SliceInfo._Underlying *__MR_SliceInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_SliceInfo_DefaultConstruct();
        }

        /// Generated from constructor `MR::SliceInfo::SliceInfo`.
        public unsafe Const_SliceInfo(MR.Const_SliceInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfo._Underlying *__MR_SliceInfo_ConstructFromAnother(MR.SliceInfo._Underlying *_other);
            _UnderlyingPtr = __MR_SliceInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// slice information
    /// these fields will be ignored in sorting
    /// \sa SliceInfoBase
    /// Generated from class `MR::SliceInfo`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SliceInfoBase`
    /// This is the non-const half of the class.
    public class SliceInfo : Const_SliceInfo
    {
        internal unsafe SliceInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.SliceInfoBase(SliceInfo self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_UpcastTo_MR_SliceInfoBase", ExactSpelling = true)]
            extern static MR.SliceInfoBase._Underlying *__MR_SliceInfo_UpcastTo_MR_SliceInfoBase(_Underlying *_this);
            MR.SliceInfoBase ret = new(__MR_SliceInfo_UpcastTo_MR_SliceInfoBase(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// image position
        public new unsafe MR.Mut_Vector3d ImagePos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_GetMutable_imagePos", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_SliceInfo_GetMutable_imagePos(_Underlying *_this);
                return new(__MR_SliceInfo_GetMutable_imagePos(_UnderlyingPtr), is_owning: false);
            }
        }

        /// instance number
        public new unsafe ref int InstanceNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_GetMutable_instanceNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfo_GetMutable_instanceNum(_Underlying *_this);
                return ref *__MR_SliceInfo_GetMutable_instanceNum(_UnderlyingPtr);
            }
        }

        /// layer height
        public new unsafe ref double Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_GetMutable_z", ExactSpelling = true)]
                extern static double *__MR_SliceInfo_GetMutable_z(_Underlying *_this);
                return ref *__MR_SliceInfo_GetMutable_z(_UnderlyingPtr);
            }
        }

        /// file index
        public new unsafe ref int FileNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_GetMutable_fileNum", ExactSpelling = true)]
                extern static int *__MR_SliceInfo_GetMutable_fileNum(_Underlying *_this);
                return ref *__MR_SliceInfo_GetMutable_fileNum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SliceInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SliceInfo._Underlying *__MR_SliceInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_SliceInfo_DefaultConstruct();
        }

        /// Generated from constructor `MR::SliceInfo::SliceInfo`.
        public unsafe SliceInfo(MR.Const_SliceInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfo._Underlying *__MR_SliceInfo_ConstructFromAnother(MR.SliceInfo._Underlying *_other);
            _UnderlyingPtr = __MR_SliceInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SliceInfo::operator=`.
        public unsafe MR.SliceInfo Assign(MR.Const_SliceInfo _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SliceInfo_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SliceInfo._Underlying *__MR_SliceInfo_AssignFromAnother(_Underlying *_this, MR.SliceInfo._Underlying *_other);
            return new(__MR_SliceInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SliceInfo` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SliceInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SliceInfo`/`Const_SliceInfo` directly.
    public class _InOptMut_SliceInfo
    {
        public SliceInfo? Opt;

        public _InOptMut_SliceInfo() {}
        public _InOptMut_SliceInfo(SliceInfo value) {Opt = value;}
        public static implicit operator _InOptMut_SliceInfo(SliceInfo value) {return new(value);}
    }

    /// This is used for optional parameters of class `SliceInfo` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SliceInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SliceInfo`/`Const_SliceInfo` to pass it to the function.
    public class _InOptConst_SliceInfo
    {
        public Const_SliceInfo? Opt;

        public _InOptConst_SliceInfo() {}
        public _InOptConst_SliceInfo(Const_SliceInfo value) {Opt = value;}
        public static implicit operator _InOptConst_SliceInfo(Const_SliceInfo value) {return new(value);}
    }

    /// Sort scan files in given vector by given slice information
    /// Generated from function `MR::sortScansByOrder`.
    public static unsafe void SortScansByOrder(MR.Std.Vector_StdFilesystemPath scans, MR.Std.Vector_MRSliceInfo zOrder)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sortScansByOrder", ExactSpelling = true)]
        extern static void __MR_sortScansByOrder(MR.Std.Vector_StdFilesystemPath._Underlying *scans, MR.Std.Vector_MRSliceInfo._Underlying *zOrder);
        __MR_sortScansByOrder(scans._UnderlyingPtr, zOrder._UnderlyingPtr);
    }

    /// Read layer heights from given scan file names
    /// Generated from function `MR::putScanFileNameInZ`.
    public static unsafe void PutScanFileNameInZ(MR.Std.Const_Vector_StdFilesystemPath scans, MR.Std.Vector_MRSliceInfo zOrder)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_putScanFileNameInZ", ExactSpelling = true)]
        extern static void __MR_putScanFileNameInZ(MR.Std.Const_Vector_StdFilesystemPath._Underlying *scans, MR.Std.Vector_MRSliceInfo._Underlying *zOrder);
        __MR_putScanFileNameInZ(scans._UnderlyingPtr, zOrder._UnderlyingPtr);
    }

    /// Sort scan files in given vector by names (respect numbers in it)
    /// Generated from function `MR::sortScanFilesByName`.
    public static unsafe void SortScanFilesByName(MR.Std.Vector_StdFilesystemPath scans)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sortScanFilesByName", ExactSpelling = true)]
        extern static void __MR_sortScanFilesByName(MR.Std.Vector_StdFilesystemPath._Underlying *scans);
        __MR_sortScanFilesByName(scans._UnderlyingPtr);
    }
}

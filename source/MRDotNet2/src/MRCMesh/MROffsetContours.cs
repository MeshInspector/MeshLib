public static partial class MR
{
    /// Generated from class `MR::OffsetContourIndex`.
    /// This is the const half of the class.
    public class Const_OffsetContourIndex : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OffsetContourIndex(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_Destroy", ExactSpelling = true)]
            extern static void __MR_OffsetContourIndex_Destroy(_Underlying *_this);
            __MR_OffsetContourIndex_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OffsetContourIndex() {Dispose(false);}

        // -1 means unknown index
        public unsafe int ContourId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_Get_contourId", ExactSpelling = true)]
                extern static int *__MR_OffsetContourIndex_Get_contourId(_Underlying *_this);
                return *__MR_OffsetContourIndex_Get_contourId(_UnderlyingPtr);
            }
        }

        // -1 means unknown index
        public unsafe int VertId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_Get_vertId", ExactSpelling = true)]
                extern static int *__MR_OffsetContourIndex_Get_vertId(_Underlying *_this);
                return *__MR_OffsetContourIndex_Get_vertId(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OffsetContourIndex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContourIndex_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContourIndex` elementwise.
        public unsafe Const_OffsetContourIndex(int contourId, int vertId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_ConstructFrom(int contourId, int vertId);
            _UnderlyingPtr = __MR_OffsetContourIndex_ConstructFrom(contourId, vertId);
        }

        /// Generated from constructor `MR::OffsetContourIndex::OffsetContourIndex`.
        public unsafe Const_OffsetContourIndex(MR.Const_OffsetContourIndex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_ConstructFromAnother(MR.OffsetContourIndex._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContourIndex_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OffsetContourIndex::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_valid", ExactSpelling = true)]
            extern static byte __MR_OffsetContourIndex_valid(_Underlying *_this);
            return __MR_OffsetContourIndex_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::OffsetContourIndex`.
    /// This is the non-const half of the class.
    public class OffsetContourIndex : Const_OffsetContourIndex
    {
        internal unsafe OffsetContourIndex(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // -1 means unknown index
        public new unsafe ref int ContourId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_GetMutable_contourId", ExactSpelling = true)]
                extern static int *__MR_OffsetContourIndex_GetMutable_contourId(_Underlying *_this);
                return ref *__MR_OffsetContourIndex_GetMutable_contourId(_UnderlyingPtr);
            }
        }

        // -1 means unknown index
        public new unsafe ref int VertId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_GetMutable_vertId", ExactSpelling = true)]
                extern static int *__MR_OffsetContourIndex_GetMutable_vertId(_Underlying *_this);
                return ref *__MR_OffsetContourIndex_GetMutable_vertId(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OffsetContourIndex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContourIndex_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContourIndex` elementwise.
        public unsafe OffsetContourIndex(int contourId, int vertId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_ConstructFrom(int contourId, int vertId);
            _UnderlyingPtr = __MR_OffsetContourIndex_ConstructFrom(contourId, vertId);
        }

        /// Generated from constructor `MR::OffsetContourIndex::OffsetContourIndex`.
        public unsafe OffsetContourIndex(MR.Const_OffsetContourIndex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_ConstructFromAnother(MR.OffsetContourIndex._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContourIndex_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OffsetContourIndex::operator=`.
        public unsafe MR.OffsetContourIndex Assign(MR.Const_OffsetContourIndex _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContourIndex_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContourIndex_AssignFromAnother(_Underlying *_this, MR.OffsetContourIndex._Underlying *_other);
            return new(__MR_OffsetContourIndex_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OffsetContourIndex` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OffsetContourIndex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContourIndex`/`Const_OffsetContourIndex` directly.
    public class _InOptMut_OffsetContourIndex
    {
        public OffsetContourIndex? Opt;

        public _InOptMut_OffsetContourIndex() {}
        public _InOptMut_OffsetContourIndex(OffsetContourIndex value) {Opt = value;}
        public static implicit operator _InOptMut_OffsetContourIndex(OffsetContourIndex value) {return new(value);}
    }

    /// This is used for optional parameters of class `OffsetContourIndex` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OffsetContourIndex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContourIndex`/`Const_OffsetContourIndex` to pass it to the function.
    public class _InOptConst_OffsetContourIndex
    {
        public Const_OffsetContourIndex? Opt;

        public _InOptConst_OffsetContourIndex() {}
        public _InOptConst_OffsetContourIndex(Const_OffsetContourIndex value) {Opt = value;}
        public static implicit operator _InOptConst_OffsetContourIndex(Const_OffsetContourIndex value) {return new(value);}
    }

    /// Generated from class `MR::OffsetContoursOrigins`.
    /// This is the const half of the class.
    public class Const_OffsetContoursOrigins : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OffsetContoursOrigins(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Destroy", ExactSpelling = true)]
            extern static void __MR_OffsetContoursOrigins_Destroy(_Underlying *_this);
            __MR_OffsetContoursOrigins_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OffsetContoursOrigins() {Dispose(false);}

        // Should be always valid
        // index of lower corresponding origin point on input contour
        public unsafe MR.Const_OffsetContourIndex LOrg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_lOrg", ExactSpelling = true)]
                extern static MR.Const_OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_Get_lOrg(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_Get_lOrg(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of lower corresponding destination point on input contour
        public unsafe MR.Const_OffsetContourIndex LDest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_lDest", ExactSpelling = true)]
                extern static MR.Const_OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_Get_lDest(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_Get_lDest(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of upper corresponding origin point on input contour
        public unsafe MR.Const_OffsetContourIndex UOrg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_uOrg", ExactSpelling = true)]
                extern static MR.Const_OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_Get_uOrg(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_Get_uOrg(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of upper corresponding destination point on input contour
        public unsafe MR.Const_OffsetContourIndex UDest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_uDest", ExactSpelling = true)]
                extern static MR.Const_OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_Get_uDest(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_Get_uDest(_UnderlyingPtr), is_owning: false);
            }
        }

        // ratio of intersection point on lOrg->lDest segment
        // 0.0 -> lOrg
        // 1.0 -> lDest
        public unsafe float LRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_lRatio", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursOrigins_Get_lRatio(_Underlying *_this);
                return *__MR_OffsetContoursOrigins_Get_lRatio(_UnderlyingPtr);
            }
        }

        // ratio of intersection point on uOrg->uDest segment
        // 0.0 -> uOrg
        // 1.0 -> uDest
        public unsafe float URatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_Get_uRatio", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursOrigins_Get_uRatio(_Underlying *_this);
                return *__MR_OffsetContoursOrigins_Get_uRatio(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OffsetContoursOrigins() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursOrigins_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursOrigins` elementwise.
        public unsafe Const_OffsetContoursOrigins(MR.Const_OffsetContourIndex lOrg, MR.Const_OffsetContourIndex lDest, MR.Const_OffsetContourIndex uOrg, MR.Const_OffsetContourIndex uDest, float lRatio, float uRatio) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_ConstructFrom(MR.OffsetContourIndex._Underlying *lOrg, MR.OffsetContourIndex._Underlying *lDest, MR.OffsetContourIndex._Underlying *uOrg, MR.OffsetContourIndex._Underlying *uDest, float lRatio, float uRatio);
            _UnderlyingPtr = __MR_OffsetContoursOrigins_ConstructFrom(lOrg._UnderlyingPtr, lDest._UnderlyingPtr, uOrg._UnderlyingPtr, uDest._UnderlyingPtr, lRatio, uRatio);
        }

        /// Generated from constructor `MR::OffsetContoursOrigins::OffsetContoursOrigins`.
        public unsafe Const_OffsetContoursOrigins(MR.Const_OffsetContoursOrigins _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_ConstructFromAnother(MR.OffsetContoursOrigins._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursOrigins_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OffsetContoursOrigins::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_valid", ExactSpelling = true)]
            extern static byte __MR_OffsetContoursOrigins_valid(_Underlying *_this);
            return __MR_OffsetContoursOrigins_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::OffsetContoursOrigins::isIntersection`.
        public unsafe bool IsIntersection()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_isIntersection", ExactSpelling = true)]
            extern static byte __MR_OffsetContoursOrigins_isIntersection(_Underlying *_this);
            return __MR_OffsetContoursOrigins_isIntersection(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::OffsetContoursOrigins`.
    /// This is the non-const half of the class.
    public class OffsetContoursOrigins : Const_OffsetContoursOrigins
    {
        internal unsafe OffsetContoursOrigins(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Should be always valid
        // index of lower corresponding origin point on input contour
        public new unsafe MR.OffsetContourIndex LOrg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_lOrg", ExactSpelling = true)]
                extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_GetMutable_lOrg(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_GetMutable_lOrg(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of lower corresponding destination point on input contour
        public new unsafe MR.OffsetContourIndex LDest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_lDest", ExactSpelling = true)]
                extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_GetMutable_lDest(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_GetMutable_lDest(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of upper corresponding origin point on input contour
        public new unsafe MR.OffsetContourIndex UOrg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_uOrg", ExactSpelling = true)]
                extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_GetMutable_uOrg(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_GetMutable_uOrg(_UnderlyingPtr), is_owning: false);
            }
        }

        // index of upper corresponding destination point on input contour
        public new unsafe MR.OffsetContourIndex UDest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_uDest", ExactSpelling = true)]
                extern static MR.OffsetContourIndex._Underlying *__MR_OffsetContoursOrigins_GetMutable_uDest(_Underlying *_this);
                return new(__MR_OffsetContoursOrigins_GetMutable_uDest(_UnderlyingPtr), is_owning: false);
            }
        }

        // ratio of intersection point on lOrg->lDest segment
        // 0.0 -> lOrg
        // 1.0 -> lDest
        public new unsafe ref float LRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_lRatio", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursOrigins_GetMutable_lRatio(_Underlying *_this);
                return ref *__MR_OffsetContoursOrigins_GetMutable_lRatio(_UnderlyingPtr);
            }
        }

        // ratio of intersection point on uOrg->uDest segment
        // 0.0 -> uOrg
        // 1.0 -> uDest
        public new unsafe ref float URatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_GetMutable_uRatio", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursOrigins_GetMutable_uRatio(_Underlying *_this);
                return ref *__MR_OffsetContoursOrigins_GetMutable_uRatio(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OffsetContoursOrigins() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursOrigins_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursOrigins` elementwise.
        public unsafe OffsetContoursOrigins(MR.Const_OffsetContourIndex lOrg, MR.Const_OffsetContourIndex lDest, MR.Const_OffsetContourIndex uOrg, MR.Const_OffsetContourIndex uDest, float lRatio, float uRatio) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_ConstructFrom(MR.OffsetContourIndex._Underlying *lOrg, MR.OffsetContourIndex._Underlying *lDest, MR.OffsetContourIndex._Underlying *uOrg, MR.OffsetContourIndex._Underlying *uDest, float lRatio, float uRatio);
            _UnderlyingPtr = __MR_OffsetContoursOrigins_ConstructFrom(lOrg._UnderlyingPtr, lDest._UnderlyingPtr, uOrg._UnderlyingPtr, uDest._UnderlyingPtr, lRatio, uRatio);
        }

        /// Generated from constructor `MR::OffsetContoursOrigins::OffsetContoursOrigins`.
        public unsafe OffsetContoursOrigins(MR.Const_OffsetContoursOrigins _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_ConstructFromAnother(MR.OffsetContoursOrigins._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursOrigins_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OffsetContoursOrigins::operator=`.
        public unsafe MR.OffsetContoursOrigins Assign(MR.Const_OffsetContoursOrigins _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursOrigins_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursOrigins._Underlying *__MR_OffsetContoursOrigins_AssignFromAnother(_Underlying *_this, MR.OffsetContoursOrigins._Underlying *_other);
            return new(__MR_OffsetContoursOrigins_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OffsetContoursOrigins` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OffsetContoursOrigins`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursOrigins`/`Const_OffsetContoursOrigins` directly.
    public class _InOptMut_OffsetContoursOrigins
    {
        public OffsetContoursOrigins? Opt;

        public _InOptMut_OffsetContoursOrigins() {}
        public _InOptMut_OffsetContoursOrigins(OffsetContoursOrigins value) {Opt = value;}
        public static implicit operator _InOptMut_OffsetContoursOrigins(OffsetContoursOrigins value) {return new(value);}
    }

    /// This is used for optional parameters of class `OffsetContoursOrigins` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OffsetContoursOrigins`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursOrigins`/`Const_OffsetContoursOrigins` to pass it to the function.
    public class _InOptConst_OffsetContoursOrigins
    {
        public Const_OffsetContoursOrigins? Opt;

        public _InOptConst_OffsetContoursOrigins() {}
        public _InOptConst_OffsetContoursOrigins(Const_OffsetContoursOrigins value) {Opt = value;}
        public static implicit operator _InOptConst_OffsetContoursOrigins(Const_OffsetContoursOrigins value) {return new(value);}
    }

    /// Generated from class `MR::OffsetContoursParams`.
    /// This is the const half of the class.
    public class Const_OffsetContoursParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OffsetContoursParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Destroy", ExactSpelling = true)]
            extern static void __MR_OffsetContoursParams_Destroy(_Underlying *_this);
            __MR_OffsetContoursParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OffsetContoursParams() {Dispose(false);}

        public unsafe MR.OffsetContoursParams.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_type", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.Type *__MR_OffsetContoursParams_Get_type(_Underlying *_this);
                return *__MR_OffsetContoursParams_Get_type(_UnderlyingPtr);
            }
        }

        public unsafe MR.OffsetContoursParams.EndType EndType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_endType", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.EndType *__MR_OffsetContoursParams_Get_endType(_Underlying *_this);
                return *__MR_OffsetContoursParams_Get_endType(_UnderlyingPtr);
            }
        }

        public unsafe MR.OffsetContoursParams.CornerType CornerType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_cornerType", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.CornerType *__MR_OffsetContoursParams_Get_cornerType(_Underlying *_this);
                return *__MR_OffsetContoursParams_Get_cornerType(_UnderlyingPtr);
            }
        }

        // 20 deg
        public unsafe float MinAnglePrecision
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_minAnglePrecision", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursParams_Get_minAnglePrecision(_Underlying *_this);
                return *__MR_OffsetContoursParams_Get_minAnglePrecision(_UnderlyingPtr);
            }
        }

        // 120 deg
        public unsafe float MaxSharpAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_maxSharpAngle", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursParams_Get_maxSharpAngle(_Underlying *_this);
                return *__MR_OffsetContoursParams_Get_maxSharpAngle(_UnderlyingPtr);
            }
        }

        /// optional output that maps result contour ids to input contour ids
        public unsafe ref void * IndicesMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_Get_indicesMap", ExactSpelling = true)]
                extern static void **__MR_OffsetContoursParams_Get_indicesMap(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_Get_indicesMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OffsetContoursParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursParams_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursParams` elementwise.
        public unsafe Const_OffsetContoursParams(MR.OffsetContoursParams.Type type, MR.OffsetContoursParams.EndType endType, MR.OffsetContoursParams.CornerType cornerType, float minAnglePrecision, float maxSharpAngle, MR.Std.Vector_StdVectorMROffsetContoursOrigins? indicesMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_ConstructFrom(MR.OffsetContoursParams.Type type, MR.OffsetContoursParams.EndType endType, MR.OffsetContoursParams.CornerType cornerType, float minAnglePrecision, float maxSharpAngle, MR.Std.Vector_StdVectorMROffsetContoursOrigins._Underlying *indicesMap);
            _UnderlyingPtr = __MR_OffsetContoursParams_ConstructFrom(type, endType, cornerType, minAnglePrecision, maxSharpAngle, indicesMap is not null ? indicesMap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::OffsetContoursParams::OffsetContoursParams`.
        public unsafe Const_OffsetContoursParams(MR.Const_OffsetContoursParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_ConstructFromAnother(MR.OffsetContoursParams._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// type of positive offset curve in corners
        public enum CornerType : int
        {
            ///< creates round corners (use `minAnglePrecision`)
            Round = 0,
            ///< creates sharp connected corner (use `maxSharpAngle` as limit)
            Sharp = 1,
        }

        /// type of offsetting on ends of non-closed contours
        public enum EndType : int
        {
            ///< creates round ends (use `minAnglePrecision`)
            Round = 0,
            ///< creates sharp end (same as Round with `minAnglePrecision` < 180 deg)
            Cut = 1,
        }

        /// type of offset
        public enum Type : int
        {
            ///< One-side signed offset, requires closed contours
            Offset = 0,
            ///< Two-side offset
            Shell = 1,
        }
    }

    /// Generated from class `MR::OffsetContoursParams`.
    /// This is the non-const half of the class.
    public class OffsetContoursParams : Const_OffsetContoursParams
    {
        internal unsafe OffsetContoursParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.OffsetContoursParams.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_type", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.Type *__MR_OffsetContoursParams_GetMutable_type(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_type(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.OffsetContoursParams.EndType EndType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_endType", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.EndType *__MR_OffsetContoursParams_GetMutable_endType(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_endType(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.OffsetContoursParams.CornerType CornerType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_cornerType", ExactSpelling = true)]
                extern static MR.OffsetContoursParams.CornerType *__MR_OffsetContoursParams_GetMutable_cornerType(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_cornerType(_UnderlyingPtr);
            }
        }

        // 20 deg
        public new unsafe ref float MinAnglePrecision
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_minAnglePrecision", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursParams_GetMutable_minAnglePrecision(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_minAnglePrecision(_UnderlyingPtr);
            }
        }

        // 120 deg
        public new unsafe ref float MaxSharpAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_maxSharpAngle", ExactSpelling = true)]
                extern static float *__MR_OffsetContoursParams_GetMutable_maxSharpAngle(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_maxSharpAngle(_UnderlyingPtr);
            }
        }

        /// optional output that maps result contour ids to input contour ids
        public new unsafe ref void * IndicesMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_GetMutable_indicesMap", ExactSpelling = true)]
                extern static void **__MR_OffsetContoursParams_GetMutable_indicesMap(_Underlying *_this);
                return ref *__MR_OffsetContoursParams_GetMutable_indicesMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OffsetContoursParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursParams_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursParams` elementwise.
        public unsafe OffsetContoursParams(MR.OffsetContoursParams.Type type, MR.OffsetContoursParams.EndType endType, MR.OffsetContoursParams.CornerType cornerType, float minAnglePrecision, float maxSharpAngle, MR.Std.Vector_StdVectorMROffsetContoursOrigins? indicesMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_ConstructFrom(MR.OffsetContoursParams.Type type, MR.OffsetContoursParams.EndType endType, MR.OffsetContoursParams.CornerType cornerType, float minAnglePrecision, float maxSharpAngle, MR.Std.Vector_StdVectorMROffsetContoursOrigins._Underlying *indicesMap);
            _UnderlyingPtr = __MR_OffsetContoursParams_ConstructFrom(type, endType, cornerType, minAnglePrecision, maxSharpAngle, indicesMap is not null ? indicesMap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::OffsetContoursParams::OffsetContoursParams`.
        public unsafe OffsetContoursParams(MR.Const_OffsetContoursParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_ConstructFromAnother(MR.OffsetContoursParams._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OffsetContoursParams::operator=`.
        public unsafe MR.OffsetContoursParams Assign(MR.Const_OffsetContoursParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursParams._Underlying *__MR_OffsetContoursParams_AssignFromAnother(_Underlying *_this, MR.OffsetContoursParams._Underlying *_other);
            return new(__MR_OffsetContoursParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OffsetContoursParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OffsetContoursParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursParams`/`Const_OffsetContoursParams` directly.
    public class _InOptMut_OffsetContoursParams
    {
        public OffsetContoursParams? Opt;

        public _InOptMut_OffsetContoursParams() {}
        public _InOptMut_OffsetContoursParams(OffsetContoursParams value) {Opt = value;}
        public static implicit operator _InOptMut_OffsetContoursParams(OffsetContoursParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `OffsetContoursParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OffsetContoursParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursParams`/`Const_OffsetContoursParams` to pass it to the function.
    public class _InOptConst_OffsetContoursParams
    {
        public Const_OffsetContoursParams? Opt;

        public _InOptConst_OffsetContoursParams() {}
        public _InOptConst_OffsetContoursParams(Const_OffsetContoursParams value) {Opt = value;}
        public static implicit operator _InOptConst_OffsetContoursParams(Const_OffsetContoursParams value) {return new(value);}
    }

    /// Parameters of restoring Z coordinate of XY offset 3d contours
    /// Generated from class `MR::OffsetContoursRestoreZParams`.
    /// This is the const half of the class.
    public class Const_OffsetContoursRestoreZParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OffsetContoursRestoreZParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_Destroy", ExactSpelling = true)]
            extern static void __MR_OffsetContoursRestoreZParams_Destroy(_Underlying *_this);
            __MR_OffsetContoursRestoreZParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OffsetContoursRestoreZParams() {Dispose(false);}

        public unsafe MR.Std.Const_Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef ZCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_Get_zCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef._Underlying *__MR_OffsetContoursRestoreZParams_Get_zCallback(_Underlying *_this);
                return new(__MR_OffsetContoursRestoreZParams_Get_zCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if > 0 z coordinate will be relaxed this many iterations
        public unsafe int RelaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_Get_relaxIterations", ExactSpelling = true)]
                extern static int *__MR_OffsetContoursRestoreZParams_Get_relaxIterations(_Underlying *_this);
                return *__MR_OffsetContoursRestoreZParams_Get_relaxIterations(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OffsetContoursRestoreZParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursRestoreZParams` elementwise.
        public unsafe Const_OffsetContoursRestoreZParams(MR.Std._ByValue_Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef zCallback, int relaxIterations) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_ConstructFrom(MR.Misc._PassBy zCallback_pass_by, MR.Std.Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef._Underlying *zCallback, int relaxIterations);
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_ConstructFrom(zCallback.PassByMode, zCallback.Value is not null ? zCallback.Value._UnderlyingPtr : null, relaxIterations);
        }

        /// Generated from constructor `MR::OffsetContoursRestoreZParams::OffsetContoursRestoreZParams`.
        public unsafe Const_OffsetContoursRestoreZParams(MR._ByValue_OffsetContoursRestoreZParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OffsetContoursRestoreZParams._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Parameters of restoring Z coordinate of XY offset 3d contours
    /// Generated from class `MR::OffsetContoursRestoreZParams`.
    /// This is the non-const half of the class.
    public class OffsetContoursRestoreZParams : Const_OffsetContoursRestoreZParams
    {
        internal unsafe OffsetContoursRestoreZParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef ZCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_GetMutable_zCallback", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef._Underlying *__MR_OffsetContoursRestoreZParams_GetMutable_zCallback(_Underlying *_this);
                return new(__MR_OffsetContoursRestoreZParams_GetMutable_zCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if > 0 z coordinate will be relaxed this many iterations
        public new unsafe ref int RelaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_GetMutable_relaxIterations", ExactSpelling = true)]
                extern static int *__MR_OffsetContoursRestoreZParams_GetMutable_relaxIterations(_Underlying *_this);
                return ref *__MR_OffsetContoursRestoreZParams_GetMutable_relaxIterations(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OffsetContoursRestoreZParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_DefaultConstruct();
        }

        /// Constructs `MR::OffsetContoursRestoreZParams` elementwise.
        public unsafe OffsetContoursRestoreZParams(MR.Std._ByValue_Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef zCallback, int relaxIterations) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_ConstructFrom(MR.Misc._PassBy zCallback_pass_by, MR.Std.Function_FloatFuncFromConstStdVectorStdVectorMRVector2fRefConstMROffsetContourIndexRefConstMROffsetContoursOriginsRef._Underlying *zCallback, int relaxIterations);
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_ConstructFrom(zCallback.PassByMode, zCallback.Value is not null ? zCallback.Value._UnderlyingPtr : null, relaxIterations);
        }

        /// Generated from constructor `MR::OffsetContoursRestoreZParams::OffsetContoursRestoreZParams`.
        public unsafe OffsetContoursRestoreZParams(MR._ByValue_OffsetContoursRestoreZParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OffsetContoursRestoreZParams._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetContoursRestoreZParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::OffsetContoursRestoreZParams::operator=`.
        public unsafe MR.OffsetContoursRestoreZParams Assign(MR._ByValue_OffsetContoursRestoreZParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetContoursRestoreZParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OffsetContoursRestoreZParams._Underlying *__MR_OffsetContoursRestoreZParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.OffsetContoursRestoreZParams._Underlying *_other);
            return new(__MR_OffsetContoursRestoreZParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `OffsetContoursRestoreZParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `OffsetContoursRestoreZParams`/`Const_OffsetContoursRestoreZParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_OffsetContoursRestoreZParams
    {
        internal readonly Const_OffsetContoursRestoreZParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_OffsetContoursRestoreZParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_OffsetContoursRestoreZParams(Const_OffsetContoursRestoreZParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_OffsetContoursRestoreZParams(Const_OffsetContoursRestoreZParams arg) {return new(arg);}
        public _ByValue_OffsetContoursRestoreZParams(MR.Misc._Moved<OffsetContoursRestoreZParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_OffsetContoursRestoreZParams(MR.Misc._Moved<OffsetContoursRestoreZParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `OffsetContoursRestoreZParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OffsetContoursRestoreZParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursRestoreZParams`/`Const_OffsetContoursRestoreZParams` directly.
    public class _InOptMut_OffsetContoursRestoreZParams
    {
        public OffsetContoursRestoreZParams? Opt;

        public _InOptMut_OffsetContoursRestoreZParams() {}
        public _InOptMut_OffsetContoursRestoreZParams(OffsetContoursRestoreZParams value) {Opt = value;}
        public static implicit operator _InOptMut_OffsetContoursRestoreZParams(OffsetContoursRestoreZParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `OffsetContoursRestoreZParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OffsetContoursRestoreZParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetContoursRestoreZParams`/`Const_OffsetContoursRestoreZParams` to pass it to the function.
    public class _InOptConst_OffsetContoursRestoreZParams
    {
        public Const_OffsetContoursRestoreZParams? Opt;

        public _InOptConst_OffsetContoursRestoreZParams() {}
        public _InOptConst_OffsetContoursRestoreZParams(Const_OffsetContoursRestoreZParams value) {Opt = value;}
        public static implicit operator _InOptConst_OffsetContoursRestoreZParams(Const_OffsetContoursRestoreZParams value) {return new(value);}
    }

    /// offsets 2d contours in plane
    /// Generated from function `MR::offsetContours`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> OffsetContours(MR.Std.Const_Vector_StdVectorMRVector2f contours, float offset, MR.Const_OffsetContoursParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetContours_3_float", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_offsetContours_3_float(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, float offset, MR.Const_OffsetContoursParams._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_offsetContours_3_float(contours._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// offsets 2d contours in plane
    /// Generated from function `MR::offsetContours`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> OffsetContours(MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.Std._ByValue_Function_FloatFuncFromIntInt offset, MR.Const_OffsetContoursParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetContours_3_std_function_float_func_from_int_int", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_offsetContours_3_std_function_float_func_from_int_int(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.Misc._PassBy offset_pass_by, MR.Std.Function_FloatFuncFromIntInt._Underlying *offset, MR.Const_OffsetContoursParams._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_offsetContours_3_std_function_float_func_from_int_int(contours._UnderlyingPtr, offset.PassByMode, offset.Value is not null ? offset.Value._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// offsets 3d contours in XY plane
    /// Generated from function `MR::offsetContours`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `zParmas` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> OffsetContours(MR.Std.Const_Vector_StdVectorMRVector3f contours, float offset, MR.Const_OffsetContoursParams? params_ = null, MR.Const_OffsetContoursRestoreZParams? zParmas = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetContours_4_float", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_offsetContours_4_float(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours, float offset, MR.Const_OffsetContoursParams._Underlying *params_, MR.Const_OffsetContoursRestoreZParams._Underlying *zParmas);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_offsetContours_4_float(contours._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null, zParmas is not null ? zParmas._UnderlyingPtr : null), is_owning: true));
    }

    /// offsets 3d contours in XY plane
    /// Generated from function `MR::offsetContours`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `zParmas` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> OffsetContours(MR.Std.Const_Vector_StdVectorMRVector3f contours, MR.Std._ByValue_Function_FloatFuncFromIntInt offset, MR.Const_OffsetContoursParams? params_ = null, MR.Const_OffsetContoursRestoreZParams? zParmas = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetContours_4_std_function_float_func_from_int_int", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_offsetContours_4_std_function_float_func_from_int_int(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours, MR.Misc._PassBy offset_pass_by, MR.Std.Function_FloatFuncFromIntInt._Underlying *offset, MR.Const_OffsetContoursParams._Underlying *params_, MR.Const_OffsetContoursRestoreZParams._Underlying *zParmas);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_offsetContours_4_std_function_float_func_from_int_int(contours._UnderlyingPtr, offset.PassByMode, offset.Value is not null ? offset.Value._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null, zParmas is not null ? zParmas._UnderlyingPtr : null), is_owning: true));
    }
}

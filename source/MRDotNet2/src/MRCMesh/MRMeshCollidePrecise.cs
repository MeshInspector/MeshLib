public static partial class MR
{
    /// edge from one mesh and triangle from another mesh
    /// Generated from class `MR::EdgeTri`.
    /// This is the const half of the class.
    public class Const_EdgeTri : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_EdgeTri>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgeTri(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeTri_Destroy(_Underlying *_this);
            __MR_EdgeTri_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeTri() {Dispose(false);}

        public unsafe MR.Const_EdgeId Edge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_Get_edge", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_EdgeTri_Get_edge(_Underlying *_this);
                return new(__MR_EdgeTri_Get_edge(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceId Tri
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_Get_tri", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_EdgeTri_Get_tri(_Underlying *_this);
                return new(__MR_EdgeTri_Get_tri(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeTri() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeTri_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeTri::EdgeTri`.
        public unsafe Const_EdgeTri(MR.Const_EdgeTri _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_ConstructFromAnother(MR.EdgeTri._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeTri_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgeTri::EdgeTri`.
        public unsafe Const_EdgeTri(MR.EdgeId e, MR.FaceId t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_Construct", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_Construct(MR.EdgeId e, MR.FaceId t);
            _UnderlyingPtr = __MR_EdgeTri_Construct(e, t);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_EdgeTri a, MR.Const_EdgeTri b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgeTri", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgeTri(MR.Const_EdgeTri._Underlying *a, MR.Const_EdgeTri._Underlying *b);
            return __MR_equal_MR_EdgeTri(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_EdgeTri a, MR.Const_EdgeTri b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_EdgeTri? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_EdgeTri)
                return this == (MR.Const_EdgeTri)other;
            return false;
        }
    }

    /// edge from one mesh and triangle from another mesh
    /// Generated from class `MR::EdgeTri`.
    /// This is the non-const half of the class.
    public class EdgeTri : Const_EdgeTri
    {
        internal unsafe EdgeTri(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_EdgeId Edge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_GetMutable_edge", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_EdgeTri_GetMutable_edge(_Underlying *_this);
                return new(__MR_EdgeTri_GetMutable_edge(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_FaceId Tri
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_GetMutable_tri", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_EdgeTri_GetMutable_tri(_Underlying *_this);
                return new(__MR_EdgeTri_GetMutable_tri(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeTri() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeTri_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeTri::EdgeTri`.
        public unsafe EdgeTri(MR.Const_EdgeTri _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_ConstructFromAnother(MR.EdgeTri._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeTri_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgeTri::EdgeTri`.
        public unsafe EdgeTri(MR.EdgeId e, MR.FaceId t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_Construct", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_Construct(MR.EdgeId e, MR.FaceId t);
            _UnderlyingPtr = __MR_EdgeTri_Construct(e, t);
        }

        /// Generated from method `MR::EdgeTri::operator=`.
        public unsafe MR.EdgeTri Assign(MR.Const_EdgeTri _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeTri_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_EdgeTri_AssignFromAnother(_Underlying *_this, MR.EdgeTri._Underlying *_other);
            return new(__MR_EdgeTri_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EdgeTri` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeTri`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeTri`/`Const_EdgeTri` directly.
    public class _InOptMut_EdgeTri
    {
        public EdgeTri? Opt;

        public _InOptMut_EdgeTri() {}
        public _InOptMut_EdgeTri(EdgeTri value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeTri(EdgeTri value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgeTri` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeTri`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeTri`/`Const_EdgeTri` to pass it to the function.
    public class _InOptConst_EdgeTri
    {
        public Const_EdgeTri? Opt;

        public _InOptConst_EdgeTri() {}
        public _InOptConst_EdgeTri(Const_EdgeTri value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeTri(Const_EdgeTri value) {return new(value);}
    }

    /// if isEdgeATriB() == true,  then stores edge from mesh A and triangle from mesh B
    /// if isEdgeATriB() == false, then stores edge from mesh B and triangle from mesh A
    /// Generated from class `MR::VarEdgeTri`.
    /// This is the const half of the class.
    public class Const_VarEdgeTri : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_VarEdgeTri>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VarEdgeTri(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Destroy", ExactSpelling = true)]
            extern static void __MR_VarEdgeTri_Destroy(_Underlying *_this);
            __MR_VarEdgeTri_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VarEdgeTri() {Dispose(false);}

        public unsafe MR.Const_EdgeId Edge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Get_edge", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_VarEdgeTri_Get_edge(_Underlying *_this);
                return new(__MR_VarEdgeTri_Get_edge(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.VarEdgeTri.Const_FlaggedTri FlaggedTri_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Get_flaggedTri", ExactSpelling = true)]
                extern static MR.VarEdgeTri.Const_FlaggedTri._Underlying *__MR_VarEdgeTri_Get_flaggedTri(_Underlying *_this);
                return new(__MR_VarEdgeTri_Get_flaggedTri(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VarEdgeTri() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_DefaultConstruct();
            _UnderlyingPtr = __MR_VarEdgeTri_DefaultConstruct();
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe Const_VarEdgeTri(MR.Const_VarEdgeTri _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_ConstructFromAnother(MR.VarEdgeTri._Underlying *_other);
            _UnderlyingPtr = __MR_VarEdgeTri_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe Const_VarEdgeTri(bool isEdgeATriB, MR.EdgeId e, MR.FaceId t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Construct_3", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_Construct_3(byte isEdgeATriB, MR.EdgeId e, MR.FaceId t);
            _UnderlyingPtr = __MR_VarEdgeTri_Construct_3(isEdgeATriB ? (byte)1 : (byte)0, e, t);
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe Const_VarEdgeTri(bool isEdgeATriB, MR.Const_EdgeTri et) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Construct_2", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_Construct_2(byte isEdgeATriB, MR.Const_EdgeTri._Underlying *et);
            _UnderlyingPtr = __MR_VarEdgeTri_Construct_2(isEdgeATriB ? (byte)1 : (byte)0, et._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::VarEdgeTri::operator bool`.
        public static unsafe explicit operator bool(MR.Const_VarEdgeTri _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VarEdgeTri_ConvertTo_bool(MR.Const_VarEdgeTri._Underlying *_this);
            return __MR_VarEdgeTri_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VarEdgeTri::tri`.
        public unsafe MR.FaceId Tri()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_tri", ExactSpelling = true)]
            extern static MR.FaceId __MR_VarEdgeTri_tri(_Underlying *_this);
            return __MR_VarEdgeTri_tri(_UnderlyingPtr);
        }

        /// Generated from method `MR::VarEdgeTri::isEdgeATriB`.
        public unsafe bool IsEdgeATriB()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_isEdgeATriB", ExactSpelling = true)]
            extern static byte __MR_VarEdgeTri_isEdgeATriB(_Underlying *_this);
            return __MR_VarEdgeTri_isEdgeATriB(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VarEdgeTri::edgeTri`.
        public unsafe MR.EdgeTri EdgeTri()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_edgeTri", ExactSpelling = true)]
            extern static MR.EdgeTri._Underlying *__MR_VarEdgeTri_edgeTri(_Underlying *_this);
            return new(__MR_VarEdgeTri_edgeTri(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::VarEdgeTri::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_valid", ExactSpelling = true)]
            extern static byte __MR_VarEdgeTri_valid(_Underlying *_this);
            return __MR_VarEdgeTri_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VarEdgeTri::operator==`.
        public static unsafe bool operator==(MR.Const_VarEdgeTri _this, MR.Const_VarEdgeTri _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VarEdgeTri", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VarEdgeTri(MR.Const_VarEdgeTri._Underlying *_this, MR.Const_VarEdgeTri._Underlying *_1);
            return __MR_equal_MR_VarEdgeTri(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_VarEdgeTri _this, MR.Const_VarEdgeTri _1)
        {
            return !(_this == _1);
        }

        /// Generated from class `MR::VarEdgeTri::FlaggedTri`.
        /// This is the const half of the class.
        public class Const_FlaggedTri : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VarEdgeTri.Const_FlaggedTri>
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_FlaggedTri(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_Destroy", ExactSpelling = true)]
                extern static void __MR_VarEdgeTri_FlaggedTri_Destroy(_Underlying *_this);
                __MR_VarEdgeTri_FlaggedTri_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_FlaggedTri() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_FlaggedTri() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_FlaggedTri_DefaultConstruct();
                _UnderlyingPtr = __MR_VarEdgeTri_FlaggedTri_DefaultConstruct();
            }

            /// Generated from constructor `MR::VarEdgeTri::FlaggedTri::FlaggedTri`.
            public unsafe Const_FlaggedTri(MR.VarEdgeTri.Const_FlaggedTri _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_FlaggedTri_ConstructFromAnother(MR.VarEdgeTri.FlaggedTri._Underlying *_other);
                _UnderlyingPtr = __MR_VarEdgeTri_FlaggedTri_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::VarEdgeTri::FlaggedTri::operator==`.
            public static unsafe bool operator==(MR.VarEdgeTri.Const_FlaggedTri _this, MR.VarEdgeTri.Const_FlaggedTri _1)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VarEdgeTri_FlaggedTri", ExactSpelling = true)]
                extern static byte __MR_equal_MR_VarEdgeTri_FlaggedTri(MR.VarEdgeTri.Const_FlaggedTri._Underlying *_this, MR.VarEdgeTri.Const_FlaggedTri._Underlying *_1);
                return __MR_equal_MR_VarEdgeTri_FlaggedTri(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
            }

            public static unsafe bool operator!=(MR.VarEdgeTri.Const_FlaggedTri _this, MR.VarEdgeTri.Const_FlaggedTri _1)
            {
                return !(_this == _1);
            }

            // IEquatable:

            public bool Equals(MR.VarEdgeTri.Const_FlaggedTri? _1)
            {
                if (_1 is null)
                    return false;
                return this == _1;
            }

            public override bool Equals(object? other)
            {
                if (other is null)
                    return false;
                if (other is MR.VarEdgeTri.Const_FlaggedTri)
                    return this == (MR.VarEdgeTri.Const_FlaggedTri)other;
                return false;
            }
        }

        /// Generated from class `MR::VarEdgeTri::FlaggedTri`.
        /// This is the non-const half of the class.
        public class FlaggedTri : Const_FlaggedTri
        {
            internal unsafe FlaggedTri(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe FlaggedTri() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_FlaggedTri_DefaultConstruct();
                _UnderlyingPtr = __MR_VarEdgeTri_FlaggedTri_DefaultConstruct();
            }

            /// Generated from constructor `MR::VarEdgeTri::FlaggedTri::FlaggedTri`.
            public unsafe FlaggedTri(MR.VarEdgeTri.Const_FlaggedTri _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_FlaggedTri_ConstructFromAnother(MR.VarEdgeTri.FlaggedTri._Underlying *_other);
                _UnderlyingPtr = __MR_VarEdgeTri_FlaggedTri_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::VarEdgeTri::FlaggedTri::operator=`.
            public unsafe MR.VarEdgeTri.FlaggedTri Assign(MR.VarEdgeTri.Const_FlaggedTri _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_FlaggedTri_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_FlaggedTri_AssignFromAnother(_Underlying *_this, MR.VarEdgeTri.FlaggedTri._Underlying *_other);
                return new(__MR_VarEdgeTri_FlaggedTri_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `FlaggedTri` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_FlaggedTri`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FlaggedTri`/`Const_FlaggedTri` directly.
        public class _InOptMut_FlaggedTri
        {
            public FlaggedTri? Opt;

            public _InOptMut_FlaggedTri() {}
            public _InOptMut_FlaggedTri(FlaggedTri value) {Opt = value;}
            public static implicit operator _InOptMut_FlaggedTri(FlaggedTri value) {return new(value);}
        }

        /// This is used for optional parameters of class `FlaggedTri` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_FlaggedTri`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FlaggedTri`/`Const_FlaggedTri` to pass it to the function.
        public class _InOptConst_FlaggedTri
        {
            public Const_FlaggedTri? Opt;

            public _InOptConst_FlaggedTri() {}
            public _InOptConst_FlaggedTri(Const_FlaggedTri value) {Opt = value;}
            public static implicit operator _InOptConst_FlaggedTri(Const_FlaggedTri value) {return new(value);}
        }

        // IEquatable:

        public bool Equals(MR.Const_VarEdgeTri? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_VarEdgeTri)
                return this == (MR.Const_VarEdgeTri)other;
            return false;
        }
    }

    /// if isEdgeATriB() == true,  then stores edge from mesh A and triangle from mesh B
    /// if isEdgeATriB() == false, then stores edge from mesh B and triangle from mesh A
    /// Generated from class `MR::VarEdgeTri`.
    /// This is the non-const half of the class.
    public class VarEdgeTri : Const_VarEdgeTri
    {
        internal unsafe VarEdgeTri(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_EdgeId Edge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_GetMutable_edge", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_VarEdgeTri_GetMutable_edge(_Underlying *_this);
                return new(__MR_VarEdgeTri_GetMutable_edge(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VarEdgeTri.FlaggedTri FlaggedTri_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_GetMutable_flaggedTri", ExactSpelling = true)]
                extern static MR.VarEdgeTri.FlaggedTri._Underlying *__MR_VarEdgeTri_GetMutable_flaggedTri(_Underlying *_this);
                return new(__MR_VarEdgeTri_GetMutable_flaggedTri(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VarEdgeTri() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_DefaultConstruct();
            _UnderlyingPtr = __MR_VarEdgeTri_DefaultConstruct();
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe VarEdgeTri(MR.Const_VarEdgeTri _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_ConstructFromAnother(MR.VarEdgeTri._Underlying *_other);
            _UnderlyingPtr = __MR_VarEdgeTri_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe VarEdgeTri(bool isEdgeATriB, MR.EdgeId e, MR.FaceId t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Construct_3", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_Construct_3(byte isEdgeATriB, MR.EdgeId e, MR.FaceId t);
            _UnderlyingPtr = __MR_VarEdgeTri_Construct_3(isEdgeATriB ? (byte)1 : (byte)0, e, t);
        }

        /// Generated from constructor `MR::VarEdgeTri::VarEdgeTri`.
        public unsafe VarEdgeTri(bool isEdgeATriB, MR.Const_EdgeTri et) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_Construct_2", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_Construct_2(byte isEdgeATriB, MR.Const_EdgeTri._Underlying *et);
            _UnderlyingPtr = __MR_VarEdgeTri_Construct_2(isEdgeATriB ? (byte)1 : (byte)0, et._UnderlyingPtr);
        }

        /// Generated from method `MR::VarEdgeTri::operator=`.
        public unsafe MR.VarEdgeTri Assign(MR.Const_VarEdgeTri _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VarEdgeTri_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VarEdgeTri._Underlying *__MR_VarEdgeTri_AssignFromAnother(_Underlying *_this, MR.VarEdgeTri._Underlying *_other);
            return new(__MR_VarEdgeTri_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VarEdgeTri` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VarEdgeTri`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VarEdgeTri`/`Const_VarEdgeTri` directly.
    public class _InOptMut_VarEdgeTri
    {
        public VarEdgeTri? Opt;

        public _InOptMut_VarEdgeTri() {}
        public _InOptMut_VarEdgeTri(VarEdgeTri value) {Opt = value;}
        public static implicit operator _InOptMut_VarEdgeTri(VarEdgeTri value) {return new(value);}
    }

    /// This is used for optional parameters of class `VarEdgeTri` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VarEdgeTri`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VarEdgeTri`/`Const_VarEdgeTri` to pass it to the function.
    public class _InOptConst_VarEdgeTri
    {
        public Const_VarEdgeTri? Opt;

        public _InOptConst_VarEdgeTri() {}
        public _InOptConst_VarEdgeTri(Const_VarEdgeTri value) {Opt = value;}
        public static implicit operator _InOptConst_VarEdgeTri(Const_VarEdgeTri value) {return new(value);}
    }

    /**
    * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param anyIntersection if true then the function returns as fast as it finds any intersection
    */
    /// Generated from function `MR::findCollidingEdgeTrisPrecise`.
    /// Parameter `anyIntersection` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVarEdgeTri> FindCollidingEdgeTrisPrecise(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef conv, MR.Const_AffineXf3f? rigidB2A = null, bool? anyIntersection = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdgeTrisPrecise_5", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVarEdgeTri._Underlying *__MR_findCollidingEdgeTrisPrecise_5(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Misc._PassBy conv_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *conv, MR.Const_AffineXf3f._Underlying *rigidB2A, byte *anyIntersection);
        byte __deref_anyIntersection = anyIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_MRVarEdgeTri(__MR_findCollidingEdgeTrisPrecise_5(a._UnderlyingPtr, b._UnderlyingPtr, conv.PassByMode, conv.Value is not null ? conv.Value._UnderlyingPtr : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, anyIntersection.HasValue ? &__deref_anyIntersection : null), is_owning: true));
    }

    /**
    * \brief finds all pairs of colliding edges and triangle within one mesh
    * \param anyIntersection if true then the function returns as fast as it finds any intersection
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation, might be useful to obtain same result as in boolean operation would for mesh B
    * \param aVertsSize used in float to int conversion, might be useful to obtain same result as in boolean operation would for mesh B
    */
    /// Generated from function `MR::findSelfCollidingEdgeTrisPrecise`.
    /// Parameter `anyIntersection` defaults to `false`.
    /// Parameter `aVertSizes` defaults to `0`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeTri> FindSelfCollidingEdgeTrisPrecise(MR.Const_MeshPart mp, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef conv, bool? anyIntersection = null, MR.Const_AffineXf3f? rigidB2A = null, int? aVertSizes = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingEdgeTrisPrecise", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeTri._Underlying *__MR_findSelfCollidingEdgeTrisPrecise(MR.Const_MeshPart._Underlying *mp, MR.Misc._PassBy conv_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *conv, byte *anyIntersection, MR.Const_AffineXf3f._Underlying *rigidB2A, int *aVertSizes);
        byte __deref_anyIntersection = anyIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
        int __deref_aVertSizes = aVertSizes.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeTri(__MR_findSelfCollidingEdgeTrisPrecise(mp._UnderlyingPtr, conv.PassByMode, conv.Value is not null ? conv.Value._UnderlyingPtr : null, anyIntersection.HasValue ? &__deref_anyIntersection : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, aVertSizes.HasValue ? &__deref_aVertSizes : null), is_owning: true));
    }

    /// finds all intersections between every given edge from A and given triangles from B
    /// Generated from function `MR::findCollidingEdgeTrisPrecise`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeTri> FindCollidingEdgeTrisPrecise(MR.Const_Mesh a, MR.Std.Const_Vector_MREdgeId edgesA, MR.Const_Mesh b, MR.Std.Const_Vector_MRFaceId facesB, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef conv, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeTri._Underlying *__MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_EdgeId(MR.Const_Mesh._Underlying *a, MR.Std.Const_Vector_MREdgeId._Underlying *edgesA, MR.Const_Mesh._Underlying *b, MR.Std.Const_Vector_MRFaceId._Underlying *facesB, MR.Misc._PassBy conv_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *conv, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeTri(__MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_EdgeId(a._UnderlyingPtr, edgesA._UnderlyingPtr, b._UnderlyingPtr, facesB._UnderlyingPtr, conv.PassByMode, conv.Value is not null ? conv.Value._UnderlyingPtr : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all intersections between every given triangle from A and given edge from B
    /// Generated from function `MR::findCollidingEdgeTrisPrecise`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeTri> FindCollidingEdgeTrisPrecise(MR.Const_Mesh a, MR.Std.Const_Vector_MRFaceId facesA, MR.Const_Mesh b, MR.Std.Const_Vector_MREdgeId edgesB, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef conv, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_FaceId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeTri._Underlying *__MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_FaceId(MR.Const_Mesh._Underlying *a, MR.Std.Const_Vector_MRFaceId._Underlying *facesA, MR.Const_Mesh._Underlying *b, MR.Std.Const_Vector_MREdgeId._Underlying *edgesB, MR.Misc._PassBy conv_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *conv, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeTri(__MR_findCollidingEdgeTrisPrecise_6_std_vector_MR_FaceId(a._UnderlyingPtr, facesA._UnderlyingPtr, b._UnderlyingPtr, edgesB._UnderlyingPtr, conv.PassByMode, conv.Value is not null ? conv.Value._UnderlyingPtr : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief creates simple converters from Vector3f to Vector3i and back in mesh part area range
    */
    /// Generated from function `MR::getVectorConverters`.
    public static unsafe MR.Misc._Moved<MR.CoordinateConverters> GetVectorConverters(MR.Const_MeshPart a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVectorConverters_1", ExactSpelling = true)]
        extern static MR.CoordinateConverters._Underlying *__MR_getVectorConverters_1(MR.Const_MeshPart._Underlying *a);
        return MR.Misc.Move(new MR.CoordinateConverters(__MR_getVectorConverters_1(a._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief creates simple converters from Vector3f to Vector3i and back in mesh parts area range
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::getVectorConverters`.
    public static unsafe MR.Misc._Moved<MR.CoordinateConverters> GetVectorConverters(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVectorConverters_3", ExactSpelling = true)]
        extern static MR.CoordinateConverters._Underlying *__MR_getVectorConverters_3(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.CoordinateConverters(__MR_getVectorConverters_3(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }
}

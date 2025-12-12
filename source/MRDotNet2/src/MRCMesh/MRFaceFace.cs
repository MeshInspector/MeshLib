public static partial class MR
{
    /// a pair of faces
    /// Generated from class `MR::FaceFace`.
    /// This is the const half of the class.
    public class Const_FaceFace : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceFace(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceFace_Destroy(_Underlying *_this);
            __MR_FaceFace_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceFace() {Dispose(false);}

        public unsafe MR.Const_FaceId AFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_Get_aFace", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_FaceFace_Get_aFace(_Underlying *_this);
                return new(__MR_FaceFace_Get_aFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceId BFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_Get_bFace", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_FaceFace_Get_bFace(_Underlying *_this);
                return new(__MR_FaceFace_Get_bFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceFace_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceFace::FaceFace`.
        public unsafe Const_FaceFace(MR.Const_FaceFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_ConstructFromAnother(MR.FaceFace._Underlying *_other);
            _UnderlyingPtr = __MR_FaceFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FaceFace::FaceFace`.
        public unsafe Const_FaceFace(MR.FaceId a, MR.FaceId b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_Construct", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_Construct(MR.FaceId a, MR.FaceId b);
            _UnderlyingPtr = __MR_FaceFace_Construct(a, b);
        }
    }

    /// a pair of faces
    /// Generated from class `MR::FaceFace`.
    /// This is the non-const half of the class.
    public class FaceFace : Const_FaceFace
    {
        internal unsafe FaceFace(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_FaceId AFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_GetMutable_aFace", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_FaceFace_GetMutable_aFace(_Underlying *_this);
                return new(__MR_FaceFace_GetMutable_aFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_FaceId BFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_GetMutable_bFace", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_FaceFace_GetMutable_bFace(_Underlying *_this);
                return new(__MR_FaceFace_GetMutable_bFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceFace_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceFace::FaceFace`.
        public unsafe FaceFace(MR.Const_FaceFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_ConstructFromAnother(MR.FaceFace._Underlying *_other);
            _UnderlyingPtr = __MR_FaceFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FaceFace::FaceFace`.
        public unsafe FaceFace(MR.FaceId a, MR.FaceId b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_Construct", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_Construct(MR.FaceId a, MR.FaceId b);
            _UnderlyingPtr = __MR_FaceFace_Construct(a, b);
        }

        /// Generated from method `MR::FaceFace::operator=`.
        public unsafe MR.FaceFace Assign(MR.Const_FaceFace _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFace_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceFace._Underlying *__MR_FaceFace_AssignFromAnother(_Underlying *_this, MR.FaceFace._Underlying *_other);
            return new(__MR_FaceFace_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FaceFace` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceFace`/`Const_FaceFace` directly.
    public class _InOptMut_FaceFace
    {
        public FaceFace? Opt;

        public _InOptMut_FaceFace() {}
        public _InOptMut_FaceFace(FaceFace value) {Opt = value;}
        public static implicit operator _InOptMut_FaceFace(FaceFace value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceFace` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceFace`/`Const_FaceFace` to pass it to the function.
    public class _InOptConst_FaceFace
    {
        public Const_FaceFace? Opt;

        public _InOptConst_FaceFace() {}
        public _InOptConst_FaceFace(Const_FaceFace value) {Opt = value;}
        public static implicit operator _InOptConst_FaceFace(Const_FaceFace value) {return new(value);}
    }

    /// a pair of undirected edges
    /// Generated from class `MR::UndirectedEdgeUndirectedEdge`.
    /// This is the const half of the class.
    public class Const_UndirectedEdgeUndirectedEdge : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UndirectedEdgeUndirectedEdge(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeUndirectedEdge_Destroy(_Underlying *_this);
            __MR_UndirectedEdgeUndirectedEdge_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirectedEdgeUndirectedEdge() {Dispose(false);}

        public unsafe MR.Const_UndirectedEdgeId AUndirEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_Get_aUndirEdge", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeUndirectedEdge_Get_aUndirEdge(_Underlying *_this);
                return new(__MR_UndirectedEdgeUndirectedEdge_Get_aUndirEdge(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_UndirectedEdgeId BUndirEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_Get_bUndirEdge", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeUndirectedEdge_Get_bUndirEdge(_Underlying *_this);
                return new(__MR_UndirectedEdgeUndirectedEdge_Get_bUndirEdge(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirectedEdgeUndirectedEdge() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeUndirectedEdge::UndirectedEdgeUndirectedEdge`.
        public unsafe Const_UndirectedEdgeUndirectedEdge(MR.Const_UndirectedEdgeUndirectedEdge _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother(MR.UndirectedEdgeUndirectedEdge._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::UndirectedEdgeUndirectedEdge::UndirectedEdgeUndirectedEdge`.
        public unsafe Const_UndirectedEdgeUndirectedEdge(MR.UndirectedEdgeId a, MR.UndirectedEdgeId b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_Construct(MR.UndirectedEdgeId a, MR.UndirectedEdgeId b);
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_Construct(a, b);
        }
    }

    /// a pair of undirected edges
    /// Generated from class `MR::UndirectedEdgeUndirectedEdge`.
    /// This is the non-const half of the class.
    public class UndirectedEdgeUndirectedEdge : Const_UndirectedEdgeUndirectedEdge
    {
        internal unsafe UndirectedEdgeUndirectedEdge(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_UndirectedEdgeId AUndirEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_GetMutable_aUndirEdge", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeUndirectedEdge_GetMutable_aUndirEdge(_Underlying *_this);
                return new(__MR_UndirectedEdgeUndirectedEdge_GetMutable_aUndirEdge(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_UndirectedEdgeId BUndirEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_GetMutable_bUndirEdge", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeUndirectedEdge_GetMutable_bUndirEdge(_Underlying *_this);
                return new(__MR_UndirectedEdgeUndirectedEdge_GetMutable_bUndirEdge(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirectedEdgeUndirectedEdge() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeUndirectedEdge::UndirectedEdgeUndirectedEdge`.
        public unsafe UndirectedEdgeUndirectedEdge(MR.Const_UndirectedEdgeUndirectedEdge _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother(MR.UndirectedEdgeUndirectedEdge._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::UndirectedEdgeUndirectedEdge::UndirectedEdgeUndirectedEdge`.
        public unsafe UndirectedEdgeUndirectedEdge(MR.UndirectedEdgeId a, MR.UndirectedEdgeId b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_Construct(MR.UndirectedEdgeId a, MR.UndirectedEdgeId b);
            _UnderlyingPtr = __MR_UndirectedEdgeUndirectedEdge_Construct(a, b);
        }

        /// Generated from method `MR::UndirectedEdgeUndirectedEdge::operator=`.
        public unsafe MR.UndirectedEdgeUndirectedEdge Assign(MR.Const_UndirectedEdgeUndirectedEdge _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeUndirectedEdge_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeUndirectedEdge._Underlying *__MR_UndirectedEdgeUndirectedEdge_AssignFromAnother(_Underlying *_this, MR.UndirectedEdgeUndirectedEdge._Underlying *_other);
            return new(__MR_UndirectedEdgeUndirectedEdge_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `UndirectedEdgeUndirectedEdge` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirectedEdgeUndirectedEdge`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeUndirectedEdge`/`Const_UndirectedEdgeUndirectedEdge` directly.
    public class _InOptMut_UndirectedEdgeUndirectedEdge
    {
        public UndirectedEdgeUndirectedEdge? Opt;

        public _InOptMut_UndirectedEdgeUndirectedEdge() {}
        public _InOptMut_UndirectedEdgeUndirectedEdge(UndirectedEdgeUndirectedEdge value) {Opt = value;}
        public static implicit operator _InOptMut_UndirectedEdgeUndirectedEdge(UndirectedEdgeUndirectedEdge value) {return new(value);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeUndirectedEdge` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirectedEdgeUndirectedEdge`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeUndirectedEdge`/`Const_UndirectedEdgeUndirectedEdge` to pass it to the function.
    public class _InOptConst_UndirectedEdgeUndirectedEdge
    {
        public Const_UndirectedEdgeUndirectedEdge? Opt;

        public _InOptConst_UndirectedEdgeUndirectedEdge() {}
        public _InOptConst_UndirectedEdgeUndirectedEdge(Const_UndirectedEdgeUndirectedEdge value) {Opt = value;}
        public static implicit operator _InOptConst_UndirectedEdgeUndirectedEdge(Const_UndirectedEdgeUndirectedEdge value) {return new(value);}
    }
}

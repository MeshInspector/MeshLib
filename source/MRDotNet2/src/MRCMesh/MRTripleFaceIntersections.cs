public static partial class MR
{
    /// a triple of faces
    /// Generated from class `MR::FaceFaceFace`.
    /// This is the const half of the class.
    public class Const_FaceFaceFace : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceFaceFace(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceFaceFace_Destroy(_Underlying *_this);
            __MR_FaceFaceFace_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceFaceFace() {Dispose(false);}

        public unsafe MR.Const_FaceId AFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Get_aFace", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_FaceFaceFace_Get_aFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_Get_aFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceId BFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Get_bFace", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_FaceFaceFace_Get_bFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_Get_bFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceId CFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Get_cFace", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_FaceFaceFace_Get_cFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_Get_cFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceFaceFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceFaceFace_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceFaceFace::FaceFaceFace`.
        public unsafe Const_FaceFaceFace(MR.Const_FaceFaceFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_ConstructFromAnother(MR.FaceFaceFace._Underlying *_other);
            _UnderlyingPtr = __MR_FaceFaceFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FaceFaceFace::FaceFaceFace`.
        public unsafe Const_FaceFaceFace(MR.FaceId a, MR.FaceId b, MR.FaceId c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Construct", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_Construct(MR.FaceId a, MR.FaceId b, MR.FaceId c);
            _UnderlyingPtr = __MR_FaceFaceFace_Construct(a, b, c);
        }
    }

    /// a triple of faces
    /// Generated from class `MR::FaceFaceFace`.
    /// This is the non-const half of the class.
    public class FaceFaceFace : Const_FaceFaceFace
    {
        internal unsafe FaceFaceFace(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_FaceId AFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_GetMutable_aFace", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_FaceFaceFace_GetMutable_aFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_GetMutable_aFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_FaceId BFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_GetMutable_bFace", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_FaceFaceFace_GetMutable_bFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_GetMutable_bFace(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_FaceId CFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_GetMutable_cFace", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_FaceFaceFace_GetMutable_cFace(_Underlying *_this);
                return new(__MR_FaceFaceFace_GetMutable_cFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceFaceFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceFaceFace_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceFaceFace::FaceFaceFace`.
        public unsafe FaceFaceFace(MR.Const_FaceFaceFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_ConstructFromAnother(MR.FaceFaceFace._Underlying *_other);
            _UnderlyingPtr = __MR_FaceFaceFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FaceFaceFace::FaceFaceFace`.
        public unsafe FaceFaceFace(MR.FaceId a, MR.FaceId b, MR.FaceId c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_Construct", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_Construct(MR.FaceId a, MR.FaceId b, MR.FaceId c);
            _UnderlyingPtr = __MR_FaceFaceFace_Construct(a, b, c);
        }

        /// Generated from method `MR::FaceFaceFace::operator=`.
        public unsafe MR.FaceFaceFace Assign(MR.Const_FaceFaceFace _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceFaceFace_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceFaceFace._Underlying *__MR_FaceFaceFace_AssignFromAnother(_Underlying *_this, MR.FaceFaceFace._Underlying *_other);
            return new(__MR_FaceFaceFace_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FaceFaceFace` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceFaceFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceFaceFace`/`Const_FaceFaceFace` directly.
    public class _InOptMut_FaceFaceFace
    {
        public FaceFaceFace? Opt;

        public _InOptMut_FaceFaceFace() {}
        public _InOptMut_FaceFaceFace(FaceFaceFace value) {Opt = value;}
        public static implicit operator _InOptMut_FaceFaceFace(FaceFaceFace value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceFaceFace` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceFaceFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceFaceFace`/`Const_FaceFaceFace` to pass it to the function.
    public class _InOptConst_FaceFaceFace
    {
        public Const_FaceFaceFace? Opt;

        public _InOptConst_FaceFaceFace() {}
        public _InOptConst_FaceFaceFace(Const_FaceFaceFace value) {Opt = value;}
        public static implicit operator _InOptConst_FaceFaceFace(Const_FaceFaceFace value) {return new(value);}
    }

    /// given all self-intersection contours on a mesh, finds all intersecting triangle triples (every two triangles from a triple intersect);
    /// please note that not all such triples will have a common point
    /// Generated from function `MR::findTripleFaceIntersections`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceFaceFace> FindTripleFaceIntersections(MR.Const_MeshTopology topology, MR.Std.Const_Vector_StdVectorMRVarEdgeTri selfContours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTripleFaceIntersections", ExactSpelling = true)]
        extern static MR.Std.Vector_MRFaceFaceFace._Underlying *__MR_findTripleFaceIntersections(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *selfContours);
        return MR.Misc.Move(new MR.Std.Vector_MRFaceFaceFace(__MR_findTripleFaceIntersections(topology._UnderlyingPtr, selfContours._UnderlyingPtr), is_owning: true));
    }
}

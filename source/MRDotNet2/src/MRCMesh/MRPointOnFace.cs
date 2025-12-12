public static partial class MR
{
    /// a point located on some mesh's face
    /// Generated from class `MR::PointOnFace`.
    /// This is the const half of the class.
    public class Const_PointOnFace : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointOnFace(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_Destroy", ExactSpelling = true)]
            extern static void __MR_PointOnFace_Destroy(_Underlying *_this);
            __MR_PointOnFace_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointOnFace() {Dispose(false);}

        /// mesh's face containing the point
        public unsafe MR.Const_FaceId Face
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_Get_face", ExactSpelling = true)]
                extern static MR.Const_FaceId._Underlying *__MR_PointOnFace_Get_face(_Underlying *_this);
                return new(__MR_PointOnFace_Get_face(_UnderlyingPtr), is_owning: false);
            }
        }

        /// a point of the mesh's face
        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointOnFace_Get_point(_Underlying *_this);
                return new(__MR_PointOnFace_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointOnFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_DefaultConstruct();
            _UnderlyingPtr = __MR_PointOnFace_DefaultConstruct();
        }

        /// Constructs `MR::PointOnFace` elementwise.
        public unsafe Const_PointOnFace(MR.FaceId face, MR.Vector3f point) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_ConstructFrom(MR.FaceId face, MR.Vector3f point);
            _UnderlyingPtr = __MR_PointOnFace_ConstructFrom(face, point);
        }

        /// Generated from constructor `MR::PointOnFace::PointOnFace`.
        public unsafe Const_PointOnFace(MR.Const_PointOnFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_ConstructFromAnother(MR.PointOnFace._Underlying *_other);
            _UnderlyingPtr = __MR_PointOnFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PointOnFace::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PointOnFace _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PointOnFace_ConvertTo_bool(MR.Const_PointOnFace._Underlying *_this);
            return __MR_PointOnFace_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise the point is not defined
        /// Generated from method `MR::PointOnFace::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_valid", ExactSpelling = true)]
            extern static byte __MR_PointOnFace_valid(_Underlying *_this);
            return __MR_PointOnFace_valid(_UnderlyingPtr) != 0;
        }
    }

    /// a point located on some mesh's face
    /// Generated from class `MR::PointOnFace`.
    /// This is the non-const half of the class.
    public class PointOnFace : Const_PointOnFace
    {
        internal unsafe PointOnFace(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// mesh's face containing the point
        public new unsafe MR.Mut_FaceId Face
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_GetMutable_face", ExactSpelling = true)]
                extern static MR.Mut_FaceId._Underlying *__MR_PointOnFace_GetMutable_face(_Underlying *_this);
                return new(__MR_PointOnFace_GetMutable_face(_UnderlyingPtr), is_owning: false);
            }
        }

        /// a point of the mesh's face
        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointOnFace_GetMutable_point(_Underlying *_this);
                return new(__MR_PointOnFace_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointOnFace() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_DefaultConstruct();
            _UnderlyingPtr = __MR_PointOnFace_DefaultConstruct();
        }

        /// Constructs `MR::PointOnFace` elementwise.
        public unsafe PointOnFace(MR.FaceId face, MR.Vector3f point) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_ConstructFrom(MR.FaceId face, MR.Vector3f point);
            _UnderlyingPtr = __MR_PointOnFace_ConstructFrom(face, point);
        }

        /// Generated from constructor `MR::PointOnFace::PointOnFace`.
        public unsafe PointOnFace(MR.Const_PointOnFace _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_ConstructFromAnother(MR.PointOnFace._Underlying *_other);
            _UnderlyingPtr = __MR_PointOnFace_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointOnFace::operator=`.
        public unsafe MR.PointOnFace Assign(MR.Const_PointOnFace _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnFace_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnFace_AssignFromAnother(_Underlying *_this, MR.PointOnFace._Underlying *_other);
            return new(__MR_PointOnFace_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointOnFace` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointOnFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointOnFace`/`Const_PointOnFace` directly.
    public class _InOptMut_PointOnFace
    {
        public PointOnFace? Opt;

        public _InOptMut_PointOnFace() {}
        public _InOptMut_PointOnFace(PointOnFace value) {Opt = value;}
        public static implicit operator _InOptMut_PointOnFace(PointOnFace value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointOnFace` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointOnFace`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointOnFace`/`Const_PointOnFace` to pass it to the function.
    public class _InOptConst_PointOnFace
    {
        public Const_PointOnFace? Opt;

        public _InOptConst_PointOnFace() {}
        public _InOptConst_PointOnFace(Const_PointOnFace value) {Opt = value;}
        public static implicit operator _InOptConst_PointOnFace(Const_PointOnFace value) {return new(value);}
    }
}

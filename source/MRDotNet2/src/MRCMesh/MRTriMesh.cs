public static partial class MR
{
    /// very simple structure for storing mesh of triangles only,
    /// without easy navigation between neighbor elements as in Mesh
    /// Generated from class `MR::TriMesh`.
    /// This is the const half of the class.
    public class Const_TriMesh : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_Destroy", ExactSpelling = true)]
            extern static void __MR_TriMesh_Destroy(_Underlying *_this);
            __MR_TriMesh_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriMesh() {Dispose(false);}

        public unsafe MR.Const_Triangulation Tris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_Get_tris", ExactSpelling = true)]
                extern static MR.Const_Triangulation._Underlying *__MR_TriMesh_Get_tris(_Underlying *_this);
                return new(__MR_TriMesh_Get_tris(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_Get_points", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_TriMesh_Get_points(_Underlying *_this);
                return new(__MR_TriMesh_Get_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TriMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_TriMesh_DefaultConstruct();
        }

        /// Constructs `MR::TriMesh` elementwise.
        public unsafe Const_TriMesh(MR._ByValue_Triangulation tris, MR._ByValue_VertCoords points) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_ConstructFrom(MR.Misc._PassBy tris_pass_by, MR.Triangulation._Underlying *tris, MR.Misc._PassBy points_pass_by, MR.VertCoords._Underlying *points);
            _UnderlyingPtr = __MR_TriMesh_ConstructFrom(tris.PassByMode, tris.Value is not null ? tris.Value._UnderlyingPtr : null, points.PassByMode, points.Value is not null ? points.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TriMesh::TriMesh`.
        public unsafe Const_TriMesh(MR._ByValue_TriMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TriMesh._Underlying *_other);
            _UnderlyingPtr = __MR_TriMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// very simple structure for storing mesh of triangles only,
    /// without easy navigation between neighbor elements as in Mesh
    /// Generated from class `MR::TriMesh`.
    /// This is the non-const half of the class.
    public class TriMesh : Const_TriMesh
    {
        internal unsafe TriMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Triangulation Tris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_GetMutable_tris", ExactSpelling = true)]
                extern static MR.Triangulation._Underlying *__MR_TriMesh_GetMutable_tris(_Underlying *_this);
                return new(__MR_TriMesh_GetMutable_tris(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_GetMutable_points", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_TriMesh_GetMutable_points(_Underlying *_this);
                return new(__MR_TriMesh_GetMutable_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TriMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_TriMesh_DefaultConstruct();
        }

        /// Constructs `MR::TriMesh` elementwise.
        public unsafe TriMesh(MR._ByValue_Triangulation tris, MR._ByValue_VertCoords points) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_ConstructFrom(MR.Misc._PassBy tris_pass_by, MR.Triangulation._Underlying *tris, MR.Misc._PassBy points_pass_by, MR.VertCoords._Underlying *points);
            _UnderlyingPtr = __MR_TriMesh_ConstructFrom(tris.PassByMode, tris.Value is not null ? tris.Value._UnderlyingPtr : null, points.PassByMode, points.Value is not null ? points.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TriMesh::TriMesh`.
        public unsafe TriMesh(MR._ByValue_TriMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TriMesh._Underlying *_other);
            _UnderlyingPtr = __MR_TriMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TriMesh::operator=`.
        public unsafe MR.TriMesh Assign(MR._ByValue_TriMesh _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriMesh_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriMesh._Underlying *__MR_TriMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TriMesh._Underlying *_other);
            return new(__MR_TriMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TriMesh` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TriMesh`/`Const_TriMesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TriMesh
    {
        internal readonly Const_TriMesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TriMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TriMesh(Const_TriMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TriMesh(Const_TriMesh arg) {return new(arg);}
        public _ByValue_TriMesh(MR.Misc._Moved<TriMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TriMesh(MR.Misc._Moved<TriMesh> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TriMesh` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriMesh`/`Const_TriMesh` directly.
    public class _InOptMut_TriMesh
    {
        public TriMesh? Opt;

        public _InOptMut_TriMesh() {}
        public _InOptMut_TriMesh(TriMesh value) {Opt = value;}
        public static implicit operator _InOptMut_TriMesh(TriMesh value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriMesh` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriMesh`/`Const_TriMesh` to pass it to the function.
    public class _InOptConst_TriMesh
    {
        public Const_TriMesh? Opt;

        public _InOptConst_TriMesh() {}
        public _InOptConst_TriMesh(Const_TriMesh value) {Opt = value;}
        public static implicit operator _InOptConst_TriMesh(Const_TriMesh value) {return new(value);}
    }
}

public static partial class MR
{
    /** \struct MR::BooleanResult
    *
    * \brief Structure contain boolean result
    * 
    * This structure store result mesh of MR::boolean or some error info
    */
    /// Generated from class `MR::BooleanResult`.
    /// This is the const half of the class.
    public class Const_BooleanResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanResult_Destroy(_Underlying *_this);
            __MR_BooleanResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanResult() {Dispose(false);}

        /// Result mesh of boolean operation, if error occurred it would be empty
        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_BooleanResult_Get_mesh(_Underlying *_this);
                return new(__MR_BooleanResult_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If input contours have intersections, this face bit set presents faces of mesh `A` on which contours intersect
        public unsafe MR.Const_FaceBitSet MeshABadContourFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_Get_meshABadContourFaces", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_BooleanResult_Get_meshABadContourFaces(_Underlying *_this);
                return new(__MR_BooleanResult_Get_meshABadContourFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If input contours have intersections, this face bit set presents faces of mesh `B` on which contours intersect
        public unsafe MR.Const_FaceBitSet MeshBBadContourFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_Get_meshBBadContourFaces", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_BooleanResult_Get_meshBBadContourFaces(_Underlying *_this);
                return new(__MR_BooleanResult_Get_meshBBadContourFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Holds error message, empty if boolean succeed
        public unsafe MR.Std.Const_String ErrorString
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_Get_errorString", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_BooleanResult_Get_errorString(_Underlying *_this);
                return new(__MR_BooleanResult_Get_errorString(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResult_DefaultConstruct();
        }

        /// Constructs `MR::BooleanResult` elementwise.
        public unsafe Const_BooleanResult(MR._ByValue_Mesh mesh, MR._ByValue_FaceBitSet meshABadContourFaces, MR._ByValue_FaceBitSet meshBBadContourFaces, ReadOnlySpan<char> errorString) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy meshABadContourFaces_pass_by, MR.FaceBitSet._Underlying *meshABadContourFaces, MR.Misc._PassBy meshBBadContourFaces_pass_by, MR.FaceBitSet._Underlying *meshBBadContourFaces, byte *errorString, byte *errorString_end);
            byte[] __bytes_errorString = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(errorString.Length)];
            int __len_errorString = System.Text.Encoding.UTF8.GetBytes(errorString, __bytes_errorString);
            fixed (byte *__ptr_errorString = __bytes_errorString)
            {
                _UnderlyingPtr = __MR_BooleanResult_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, meshABadContourFaces.PassByMode, meshABadContourFaces.Value is not null ? meshABadContourFaces.Value._UnderlyingPtr : null, meshBBadContourFaces.PassByMode, meshBBadContourFaces.Value is not null ? meshBBadContourFaces.Value._UnderlyingPtr : null, __ptr_errorString, __ptr_errorString + __len_errorString);
            }
        }

        /// Generated from constructor `MR::BooleanResult::BooleanResult`.
        public unsafe Const_BooleanResult(MR._ByValue_BooleanResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResult._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from conversion operator `MR::BooleanResult::operator bool`.
        public static unsafe implicit operator bool(MR.Const_BooleanResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_BooleanResult_ConvertTo_bool(MR.Const_BooleanResult._Underlying *_this);
            return __MR_BooleanResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Returns true if boolean succeed, false otherwise
        /// Generated from method `MR::BooleanResult::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_valid", ExactSpelling = true)]
            extern static byte __MR_BooleanResult_valid(_Underlying *_this);
            return __MR_BooleanResult_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::BooleanResult::operator*`.
        public unsafe MR.Const_Mesh Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_const_MR_BooleanResult_ref", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_deref_const_MR_BooleanResult_ref(_Underlying *_this);
            return new(__MR_deref_const_MR_BooleanResult_ref(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BooleanResult::operator->`.
        public unsafe MR.Const_Mesh? Arrow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_arrow_const", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_BooleanResult_arrow_const(_Underlying *_this);
            var __ret = __MR_BooleanResult_arrow_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_Mesh(__ret, is_owning: false) : null;
        }
    }

    /** \struct MR::BooleanResult
    *
    * \brief Structure contain boolean result
    * 
    * This structure store result mesh of MR::boolean or some error info
    */
    /// Generated from class `MR::BooleanResult`.
    /// This is the non-const half of the class.
    public class BooleanResult : Const_BooleanResult
    {
        internal unsafe BooleanResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Result mesh of boolean operation, if error occurred it would be empty
        public new unsafe MR.Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_GetMutable_mesh", ExactSpelling = true)]
                extern static MR.Mesh._Underlying *__MR_BooleanResult_GetMutable_mesh(_Underlying *_this);
                return new(__MR_BooleanResult_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If input contours have intersections, this face bit set presents faces of mesh `A` on which contours intersect
        public new unsafe MR.FaceBitSet MeshABadContourFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_GetMutable_meshABadContourFaces", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_BooleanResult_GetMutable_meshABadContourFaces(_Underlying *_this);
                return new(__MR_BooleanResult_GetMutable_meshABadContourFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If input contours have intersections, this face bit set presents faces of mesh `B` on which contours intersect
        public new unsafe MR.FaceBitSet MeshBBadContourFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_GetMutable_meshBBadContourFaces", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_BooleanResult_GetMutable_meshBBadContourFaces(_Underlying *_this);
                return new(__MR_BooleanResult_GetMutable_meshBBadContourFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Holds error message, empty if boolean succeed
        public new unsafe MR.Std.String ErrorString
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_GetMutable_errorString", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_BooleanResult_GetMutable_errorString(_Underlying *_this);
                return new(__MR_BooleanResult_GetMutable_errorString(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResult_DefaultConstruct();
        }

        /// Constructs `MR::BooleanResult` elementwise.
        public unsafe BooleanResult(MR._ByValue_Mesh mesh, MR._ByValue_FaceBitSet meshABadContourFaces, MR._ByValue_FaceBitSet meshBBadContourFaces, ReadOnlySpan<char> errorString) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy meshABadContourFaces_pass_by, MR.FaceBitSet._Underlying *meshABadContourFaces, MR.Misc._PassBy meshBBadContourFaces_pass_by, MR.FaceBitSet._Underlying *meshBBadContourFaces, byte *errorString, byte *errorString_end);
            byte[] __bytes_errorString = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(errorString.Length)];
            int __len_errorString = System.Text.Encoding.UTF8.GetBytes(errorString, __bytes_errorString);
            fixed (byte *__ptr_errorString = __bytes_errorString)
            {
                _UnderlyingPtr = __MR_BooleanResult_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, meshABadContourFaces.PassByMode, meshABadContourFaces.Value is not null ? meshABadContourFaces.Value._UnderlyingPtr : null, meshBBadContourFaces.PassByMode, meshBBadContourFaces.Value is not null ? meshBBadContourFaces.Value._UnderlyingPtr : null, __ptr_errorString, __ptr_errorString + __len_errorString);
            }
        }

        /// Generated from constructor `MR::BooleanResult::BooleanResult`.
        public unsafe BooleanResult(MR._ByValue_BooleanResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResult._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BooleanResult::operator=`.
        public unsafe MR.BooleanResult Assign(MR._ByValue_BooleanResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResult._Underlying *__MR_BooleanResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanResult._Underlying *_other);
            return new(__MR_BooleanResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::BooleanResult::operator*`.
        public unsafe new MR.Mesh Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_BooleanResult_ref", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_deref_MR_BooleanResult_ref(_Underlying *_this);
            return new(__MR_deref_MR_BooleanResult_ref(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BooleanResult::operator->`.
        public unsafe new MR.Mesh? Arrow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResult_arrow", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_BooleanResult_arrow(_Underlying *_this);
            var __ret = __MR_BooleanResult_arrow(_UnderlyingPtr);
            return __ret is not null ? new MR.Mesh(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `BooleanResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BooleanResult`/`Const_BooleanResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BooleanResult
    {
        internal readonly Const_BooleanResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BooleanResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BooleanResult(Const_BooleanResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BooleanResult(Const_BooleanResult arg) {return new(arg);}
        public _ByValue_BooleanResult(MR.Misc._Moved<BooleanResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BooleanResult(MR.Misc._Moved<BooleanResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BooleanResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResult`/`Const_BooleanResult` directly.
    public class _InOptMut_BooleanResult
    {
        public BooleanResult? Opt;

        public _InOptMut_BooleanResult() {}
        public _InOptMut_BooleanResult(BooleanResult value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanResult(BooleanResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResult`/`Const_BooleanResult` to pass it to the function.
    public class _InOptConst_BooleanResult
    {
        public Const_BooleanResult? Opt;

        public _InOptConst_BooleanResult() {}
        public _InOptConst_BooleanResult(Const_BooleanResult value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanResult(Const_BooleanResult value) {return new(value);}
    }

    /// Generated from class `MR::BooleanPreCutResult`.
    /// This is the const half of the class.
    public class Const_BooleanPreCutResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanPreCutResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanPreCutResult_Destroy(_Underlying *_this);
            __MR_BooleanPreCutResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanPreCutResult() {Dispose(false);}

        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_BooleanPreCutResult_Get_mesh(_Underlying *_this);
                return new(__MR_BooleanPreCutResult_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Vector_MROneMeshContour Contours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_Get_contours", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MROneMeshContour._Underlying *__MR_BooleanPreCutResult_Get_contours(_Underlying *_this);
                return new(__MR_BooleanPreCutResult_Get_contours(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanPreCutResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanPreCutResult_DefaultConstruct();
        }

        /// Constructs `MR::BooleanPreCutResult` elementwise.
        public unsafe Const_BooleanPreCutResult(MR._ByValue_Mesh mesh, MR.Std._ByValue_Vector_MROneMeshContour contours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy contours_pass_by, MR.Std.Vector_MROneMeshContour._Underlying *contours);
            _UnderlyingPtr = __MR_BooleanPreCutResult_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, contours.PassByMode, contours.Value is not null ? contours.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanPreCutResult::BooleanPreCutResult`.
        public unsafe Const_BooleanPreCutResult(MR._ByValue_BooleanPreCutResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanPreCutResult._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanPreCutResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::BooleanPreCutResult`.
    /// This is the non-const half of the class.
    public class BooleanPreCutResult : Const_BooleanPreCutResult
    {
        internal unsafe BooleanPreCutResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_GetMutable_mesh", ExactSpelling = true)]
                extern static MR.Mesh._Underlying *__MR_BooleanPreCutResult_GetMutable_mesh(_Underlying *_this);
                return new(__MR_BooleanPreCutResult_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Vector_MROneMeshContour Contours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_GetMutable_contours", ExactSpelling = true)]
                extern static MR.Std.Vector_MROneMeshContour._Underlying *__MR_BooleanPreCutResult_GetMutable_contours(_Underlying *_this);
                return new(__MR_BooleanPreCutResult_GetMutable_contours(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanPreCutResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanPreCutResult_DefaultConstruct();
        }

        /// Constructs `MR::BooleanPreCutResult` elementwise.
        public unsafe BooleanPreCutResult(MR._ByValue_Mesh mesh, MR.Std._ByValue_Vector_MROneMeshContour contours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy contours_pass_by, MR.Std.Vector_MROneMeshContour._Underlying *contours);
            _UnderlyingPtr = __MR_BooleanPreCutResult_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, contours.PassByMode, contours.Value is not null ? contours.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanPreCutResult::BooleanPreCutResult`.
        public unsafe BooleanPreCutResult(MR._ByValue_BooleanPreCutResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanPreCutResult._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanPreCutResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BooleanPreCutResult::operator=`.
        public unsafe MR.BooleanPreCutResult Assign(MR._ByValue_BooleanPreCutResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanPreCutResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanPreCutResult._Underlying *__MR_BooleanPreCutResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanPreCutResult._Underlying *_other);
            return new(__MR_BooleanPreCutResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BooleanPreCutResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BooleanPreCutResult`/`Const_BooleanPreCutResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BooleanPreCutResult
    {
        internal readonly Const_BooleanPreCutResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BooleanPreCutResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BooleanPreCutResult(Const_BooleanPreCutResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BooleanPreCutResult(Const_BooleanPreCutResult arg) {return new(arg);}
        public _ByValue_BooleanPreCutResult(MR.Misc._Moved<BooleanPreCutResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BooleanPreCutResult(MR.Misc._Moved<BooleanPreCutResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BooleanPreCutResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanPreCutResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanPreCutResult`/`Const_BooleanPreCutResult` directly.
    public class _InOptMut_BooleanPreCutResult
    {
        public BooleanPreCutResult? Opt;

        public _InOptMut_BooleanPreCutResult() {}
        public _InOptMut_BooleanPreCutResult(BooleanPreCutResult value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanPreCutResult(BooleanPreCutResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanPreCutResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanPreCutResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanPreCutResult`/`Const_BooleanPreCutResult` to pass it to the function.
    public class _InOptConst_BooleanPreCutResult
    {
        public Const_BooleanPreCutResult? Opt;

        public _InOptConst_BooleanPreCutResult() {}
        public _InOptConst_BooleanPreCutResult(Const_BooleanPreCutResult value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanPreCutResult(Const_BooleanPreCutResult value) {return new(value);}
    }

    /** \struct MR::BooleanResult
    *
    * \brief Structure with parameters for boolean call
    */
    /// Generated from class `MR::BooleanParameters`.
    /// This is the const half of the class.
    public class Const_BooleanParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanParameters_Destroy(_Underlying *_this);
            __MR_BooleanParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanParameters() {Dispose(false);}

        /// Transform from mesh `B` space to mesh `A` space
        public unsafe ref readonly MR.AffineXf3f * RigidB2A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_rigidB2A", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_BooleanParameters_Get_rigidB2A(_Underlying *_this);
                return ref *__MR_BooleanParameters_Get_rigidB2A(_UnderlyingPtr);
            }
        }

        /// Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
        public unsafe ref void * Mapper
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_mapper", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_Get_mapper(_Underlying *_this);
                return ref *__MR_BooleanParameters_Get_mapper(_UnderlyingPtr);
            }
        }

        /// Optional precut output of meshA, if present - does not perform boolean and just return them
        public unsafe ref void * OutPreCutA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_outPreCutA", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_Get_outPreCutA(_Underlying *_this);
                return ref *__MR_BooleanParameters_Get_outPreCutA(_UnderlyingPtr);
            }
        }

        /// Optional precut output of meshB, if present - does not perform boolean and just return them
        public unsafe ref void * OutPreCutB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_outPreCutB", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_Get_outPreCutB(_Underlying *_this);
                return ref *__MR_BooleanParameters_Get_outPreCutB(_UnderlyingPtr);
            }
        }

        /// Optional output cut edges of booleaned meshes 
        public unsafe ref void * OutCutEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_outCutEdges", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_Get_outCutEdges(_Underlying *_this);
                return ref *__MR_BooleanParameters_Get_outCutEdges(_UnderlyingPtr);
            }
        }

        /// By default produce valid operation on disconnected components
        /// if set merge all non-intersecting components
        public unsafe bool MergeAllNonIntersectingComponents
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_mergeAllNonIntersectingComponents", ExactSpelling = true)]
                extern static bool *__MR_BooleanParameters_Get_mergeAllNonIntersectingComponents(_Underlying *_this);
                return *__MR_BooleanParameters_Get_mergeAllNonIntersectingComponents(_UnderlyingPtr);
            }
        }

        /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
        /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
        /// \warning not recommended in most cases
        public unsafe bool ForceCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_forceCut", ExactSpelling = true)]
                extern static bool *__MR_BooleanParameters_Get_forceCut(_Underlying *_this);
                return *__MR_BooleanParameters_Get_forceCut(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_BooleanParameters_Get_cb(_Underlying *_this);
                return new(__MR_BooleanParameters_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanParameters_DefaultConstruct();
        }

        /// Constructs `MR::BooleanParameters` elementwise.
        public unsafe Const_BooleanParameters(MR.Const_AffineXf3f? rigidB2A, MR.BooleanResultMapper? mapper, MR.BooleanPreCutResult? outPreCutA, MR.BooleanPreCutResult? outPreCutB, MR.Std.Vector_StdVectorMREdgeId? outCutEdges, bool mergeAllNonIntersectingComponents, bool forceCut, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_ConstructFrom(MR.Const_AffineXf3f._Underlying *rigidB2A, MR.BooleanResultMapper._Underlying *mapper, MR.BooleanPreCutResult._Underlying *outPreCutA, MR.BooleanPreCutResult._Underlying *outPreCutB, MR.Std.Vector_StdVectorMREdgeId._Underlying *outCutEdges, byte mergeAllNonIntersectingComponents, byte forceCut, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_BooleanParameters_ConstructFrom(rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, mapper is not null ? mapper._UnderlyingPtr : null, outPreCutA is not null ? outPreCutA._UnderlyingPtr : null, outPreCutB is not null ? outPreCutB._UnderlyingPtr : null, outCutEdges is not null ? outCutEdges._UnderlyingPtr : null, mergeAllNonIntersectingComponents ? (byte)1 : (byte)0, forceCut ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanParameters::BooleanParameters`.
        public unsafe Const_BooleanParameters(MR._ByValue_BooleanParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /** \struct MR::BooleanResult
    *
    * \brief Structure with parameters for boolean call
    */
    /// Generated from class `MR::BooleanParameters`.
    /// This is the non-const half of the class.
    public class BooleanParameters : Const_BooleanParameters
    {
        internal unsafe BooleanParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Transform from mesh `B` space to mesh `A` space
        public new unsafe ref readonly MR.AffineXf3f * RigidB2A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_rigidB2A", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_BooleanParameters_GetMutable_rigidB2A(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_rigidB2A(_UnderlyingPtr);
            }
        }

        /// Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
        public new unsafe ref void * Mapper
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_mapper", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_GetMutable_mapper(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_mapper(_UnderlyingPtr);
            }
        }

        /// Optional precut output of meshA, if present - does not perform boolean and just return them
        public new unsafe ref void * OutPreCutA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_outPreCutA", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_GetMutable_outPreCutA(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_outPreCutA(_UnderlyingPtr);
            }
        }

        /// Optional precut output of meshB, if present - does not perform boolean and just return them
        public new unsafe ref void * OutPreCutB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_outPreCutB", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_GetMutable_outPreCutB(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_outPreCutB(_UnderlyingPtr);
            }
        }

        /// Optional output cut edges of booleaned meshes 
        public new unsafe ref void * OutCutEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_outCutEdges", ExactSpelling = true)]
                extern static void **__MR_BooleanParameters_GetMutable_outCutEdges(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_outCutEdges(_UnderlyingPtr);
            }
        }

        /// By default produce valid operation on disconnected components
        /// if set merge all non-intersecting components
        public new unsafe ref bool MergeAllNonIntersectingComponents
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_mergeAllNonIntersectingComponents", ExactSpelling = true)]
                extern static bool *__MR_BooleanParameters_GetMutable_mergeAllNonIntersectingComponents(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_mergeAllNonIntersectingComponents(_UnderlyingPtr);
            }
        }

        /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
        /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
        /// \warning not recommended in most cases
        public new unsafe ref bool ForceCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_forceCut", ExactSpelling = true)]
                extern static bool *__MR_BooleanParameters_GetMutable_forceCut(_Underlying *_this);
                return ref *__MR_BooleanParameters_GetMutable_forceCut(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_BooleanParameters_GetMutable_cb(_Underlying *_this);
                return new(__MR_BooleanParameters_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanParameters_DefaultConstruct();
        }

        /// Constructs `MR::BooleanParameters` elementwise.
        public unsafe BooleanParameters(MR.Const_AffineXf3f? rigidB2A, MR.BooleanResultMapper? mapper, MR.BooleanPreCutResult? outPreCutA, MR.BooleanPreCutResult? outPreCutB, MR.Std.Vector_StdVectorMREdgeId? outCutEdges, bool mergeAllNonIntersectingComponents, bool forceCut, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_ConstructFrom(MR.Const_AffineXf3f._Underlying *rigidB2A, MR.BooleanResultMapper._Underlying *mapper, MR.BooleanPreCutResult._Underlying *outPreCutA, MR.BooleanPreCutResult._Underlying *outPreCutB, MR.Std.Vector_StdVectorMREdgeId._Underlying *outCutEdges, byte mergeAllNonIntersectingComponents, byte forceCut, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_BooleanParameters_ConstructFrom(rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, mapper is not null ? mapper._UnderlyingPtr : null, outPreCutA is not null ? outPreCutA._UnderlyingPtr : null, outPreCutB is not null ? outPreCutB._UnderlyingPtr : null, outCutEdges is not null ? outCutEdges._UnderlyingPtr : null, mergeAllNonIntersectingComponents ? (byte)1 : (byte)0, forceCut ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanParameters::BooleanParameters`.
        public unsafe BooleanParameters(MR._ByValue_BooleanParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BooleanParameters::operator=`.
        public unsafe MR.BooleanParameters Assign(MR._ByValue_BooleanParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanParameters._Underlying *__MR_BooleanParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanParameters._Underlying *_other);
            return new(__MR_BooleanParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BooleanParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BooleanParameters`/`Const_BooleanParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BooleanParameters
    {
        internal readonly Const_BooleanParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BooleanParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BooleanParameters(Const_BooleanParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BooleanParameters(Const_BooleanParameters arg) {return new(arg);}
        public _ByValue_BooleanParameters(MR.Misc._Moved<BooleanParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BooleanParameters(MR.Misc._Moved<BooleanParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BooleanParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanParameters`/`Const_BooleanParameters` directly.
    public class _InOptMut_BooleanParameters
    {
        public BooleanParameters? Opt;

        public _InOptMut_BooleanParameters() {}
        public _InOptMut_BooleanParameters(BooleanParameters value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanParameters(BooleanParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanParameters`/`Const_BooleanParameters` to pass it to the function.
    public class _InOptConst_BooleanParameters
    {
        public Const_BooleanParameters? Opt;

        public _InOptConst_BooleanParameters() {}
        public _InOptConst_BooleanParameters(Const_BooleanParameters value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanParameters(Const_BooleanParameters value) {return new(value);}
    }

    /// vertices and points representing mesh intersection result
    /// Generated from class `MR::BooleanResultPoints`.
    /// This is the const half of the class.
    public class Const_BooleanResultPoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanResultPoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanResultPoints_Destroy(_Underlying *_this);
            __MR_BooleanResultPoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanResultPoints() {Dispose(false);}

        public unsafe MR.Const_VertBitSet MeshAVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_Get_meshAVerts", ExactSpelling = true)]
                extern static MR.Const_VertBitSet._Underlying *__MR_BooleanResultPoints_Get_meshAVerts(_Underlying *_this);
                return new(__MR_BooleanResultPoints_Get_meshAVerts(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertBitSet MeshBVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_Get_meshBVerts", ExactSpelling = true)]
                extern static MR.Const_VertBitSet._Underlying *__MR_BooleanResultPoints_Get_meshBVerts(_Underlying *_this);
                return new(__MR_BooleanResultPoints_Get_meshBVerts(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Vector_MRVector3f IntersectionPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_Get_intersectionPoints", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_BooleanResultPoints_Get_intersectionPoints(_Underlying *_this);
                return new(__MR_BooleanResultPoints_Get_intersectionPoints(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanResultPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResultPoints_DefaultConstruct();
        }

        /// Constructs `MR::BooleanResultPoints` elementwise.
        public unsafe Const_BooleanResultPoints(MR._ByValue_VertBitSet meshAVerts, MR._ByValue_VertBitSet meshBVerts, MR.Std._ByValue_Vector_MRVector3f intersectionPoints) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_ConstructFrom(MR.Misc._PassBy meshAVerts_pass_by, MR.VertBitSet._Underlying *meshAVerts, MR.Misc._PassBy meshBVerts_pass_by, MR.VertBitSet._Underlying *meshBVerts, MR.Misc._PassBy intersectionPoints_pass_by, MR.Std.Vector_MRVector3f._Underlying *intersectionPoints);
            _UnderlyingPtr = __MR_BooleanResultPoints_ConstructFrom(meshAVerts.PassByMode, meshAVerts.Value is not null ? meshAVerts.Value._UnderlyingPtr : null, meshBVerts.PassByMode, meshBVerts.Value is not null ? meshBVerts.Value._UnderlyingPtr : null, intersectionPoints.PassByMode, intersectionPoints.Value is not null ? intersectionPoints.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanResultPoints::BooleanResultPoints`.
        public unsafe Const_BooleanResultPoints(MR._ByValue_BooleanResultPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultPoints._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResultPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// vertices and points representing mesh intersection result
    /// Generated from class `MR::BooleanResultPoints`.
    /// This is the non-const half of the class.
    public class BooleanResultPoints : Const_BooleanResultPoints
    {
        internal unsafe BooleanResultPoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.VertBitSet MeshAVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_GetMutable_meshAVerts", ExactSpelling = true)]
                extern static MR.VertBitSet._Underlying *__MR_BooleanResultPoints_GetMutable_meshAVerts(_Underlying *_this);
                return new(__MR_BooleanResultPoints_GetMutable_meshAVerts(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertBitSet MeshBVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_GetMutable_meshBVerts", ExactSpelling = true)]
                extern static MR.VertBitSet._Underlying *__MR_BooleanResultPoints_GetMutable_meshBVerts(_Underlying *_this);
                return new(__MR_BooleanResultPoints_GetMutable_meshBVerts(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Vector_MRVector3f IntersectionPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_GetMutable_intersectionPoints", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3f._Underlying *__MR_BooleanResultPoints_GetMutable_intersectionPoints(_Underlying *_this);
                return new(__MR_BooleanResultPoints_GetMutable_intersectionPoints(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanResultPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResultPoints_DefaultConstruct();
        }

        /// Constructs `MR::BooleanResultPoints` elementwise.
        public unsafe BooleanResultPoints(MR._ByValue_VertBitSet meshAVerts, MR._ByValue_VertBitSet meshBVerts, MR.Std._ByValue_Vector_MRVector3f intersectionPoints) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_ConstructFrom(MR.Misc._PassBy meshAVerts_pass_by, MR.VertBitSet._Underlying *meshAVerts, MR.Misc._PassBy meshBVerts_pass_by, MR.VertBitSet._Underlying *meshBVerts, MR.Misc._PassBy intersectionPoints_pass_by, MR.Std.Vector_MRVector3f._Underlying *intersectionPoints);
            _UnderlyingPtr = __MR_BooleanResultPoints_ConstructFrom(meshAVerts.PassByMode, meshAVerts.Value is not null ? meshAVerts.Value._UnderlyingPtr : null, meshBVerts.PassByMode, meshBVerts.Value is not null ? meshBVerts.Value._UnderlyingPtr : null, intersectionPoints.PassByMode, intersectionPoints.Value is not null ? intersectionPoints.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanResultPoints::BooleanResultPoints`.
        public unsafe BooleanResultPoints(MR._ByValue_BooleanResultPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultPoints._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResultPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BooleanResultPoints::operator=`.
        public unsafe MR.BooleanResultPoints Assign(MR._ByValue_BooleanResultPoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultPoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultPoints._Underlying *__MR_BooleanResultPoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanResultPoints._Underlying *_other);
            return new(__MR_BooleanResultPoints_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BooleanResultPoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BooleanResultPoints`/`Const_BooleanResultPoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BooleanResultPoints
    {
        internal readonly Const_BooleanResultPoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BooleanResultPoints() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BooleanResultPoints(Const_BooleanResultPoints new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BooleanResultPoints(Const_BooleanResultPoints arg) {return new(arg);}
        public _ByValue_BooleanResultPoints(MR.Misc._Moved<BooleanResultPoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BooleanResultPoints(MR.Misc._Moved<BooleanResultPoints> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BooleanResultPoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanResultPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResultPoints`/`Const_BooleanResultPoints` directly.
    public class _InOptMut_BooleanResultPoints
    {
        public BooleanResultPoints? Opt;

        public _InOptMut_BooleanResultPoints() {}
        public _InOptMut_BooleanResultPoints(BooleanResultPoints value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanResultPoints(BooleanResultPoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanResultPoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanResultPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResultPoints`/`Const_BooleanResultPoints` to pass it to the function.
    public class _InOptConst_BooleanResultPoints
    {
        public Const_BooleanResultPoints? Opt;

        public _InOptConst_BooleanResultPoints() {}
        public _InOptConst_BooleanResultPoints(Const_BooleanResultPoints value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanResultPoints(Const_BooleanResultPoints value) {return new(value);}
    }

    /** \brief Performs CSG operation on two meshes
    * 
    *
    * Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
    * \snippet cpp-examples/MeshBoolean.dox.cpp 0
    *
    * \param meshA Input mesh `A`
    * \param meshB Input mesh `B`
    * \param operation CSG operation to perform
    * \param rigidB2A Transform from mesh `B` space to mesh `A` space
    * \param mapper Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
    * 
    * \note Input meshes should have no self-intersections in intersecting zone
    * \note If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)
    */
    /// Generated from function `MR::boolean`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.BooleanResult> Boolean(MR.Const_Mesh meshA, MR.Const_Mesh meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f? rigidB2A, MR.BooleanResultMapper? mapper = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_boolean_6_const_MR_Mesh_ref", ExactSpelling = true)]
        extern static MR.BooleanResult._Underlying *__MR_boolean_6_const_MR_Mesh_ref(MR.Const_Mesh._Underlying *meshA, MR.Const_Mesh._Underlying *meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f._Underlying *rigidB2A, MR.BooleanResultMapper._Underlying *mapper, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.BooleanResult(__MR_boolean_6_const_MR_Mesh_ref(meshA._UnderlyingPtr, meshB._UnderlyingPtr, operation, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, mapper is not null ? mapper._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::boolean`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.BooleanResult> Boolean(MR.Misc._Moved<MR.Mesh> meshA, MR.Misc._Moved<MR.Mesh> meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f? rigidB2A, MR.BooleanResultMapper? mapper = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_boolean_6_MR_Mesh_rvalue_ref", ExactSpelling = true)]
        extern static MR.BooleanResult._Underlying *__MR_boolean_6_MR_Mesh_rvalue_ref(MR.Mesh._Underlying *meshA, MR.Mesh._Underlying *meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f._Underlying *rigidB2A, MR.BooleanResultMapper._Underlying *mapper, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.BooleanResult(__MR_boolean_6_MR_Mesh_rvalue_ref(meshA.Value._UnderlyingPtr, meshB.Value._UnderlyingPtr, operation, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, mapper is not null ? mapper._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::boolean`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.BooleanResult> Boolean(MR.Const_Mesh meshA, MR.Const_Mesh meshB, MR.BooleanOperation operation, MR.Const_BooleanParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_boolean_4_const_MR_Mesh_ref", ExactSpelling = true)]
        extern static MR.BooleanResult._Underlying *__MR_boolean_4_const_MR_Mesh_ref(MR.Const_Mesh._Underlying *meshA, MR.Const_Mesh._Underlying *meshB, MR.BooleanOperation operation, MR.Const_BooleanParameters._Underlying *params_);
        return MR.Misc.Move(new MR.BooleanResult(__MR_boolean_4_const_MR_Mesh_ref(meshA._UnderlyingPtr, meshB._UnderlyingPtr, operation, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::boolean`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.BooleanResult> Boolean(MR.Misc._Moved<MR.Mesh> meshA, MR.Misc._Moved<MR.Mesh> meshB, MR.BooleanOperation operation, MR.Const_BooleanParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_boolean_4_MR_Mesh_rvalue_ref", ExactSpelling = true)]
        extern static MR.BooleanResult._Underlying *__MR_boolean_4_MR_Mesh_rvalue_ref(MR.Mesh._Underlying *meshA, MR.Mesh._Underlying *meshB, MR.BooleanOperation operation, MR.Const_BooleanParameters._Underlying *params_);
        return MR.Misc.Move(new MR.BooleanResult(__MR_boolean_4_MR_Mesh_rvalue_ref(meshA.Value._UnderlyingPtr, meshB.Value._UnderlyingPtr, operation, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// performs boolean operation on mesh with itself, cutting simple intersections contours and flipping their connectivity
    /// this function is experimental and likely to change signature and/or behavior in future 
    /// Generated from function `MR::selfBoolean`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> SelfBoolean(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_selfBoolean", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_selfBoolean(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_selfBoolean(mesh._UnderlyingPtr), is_owning: true));
    }

    /// returns intersection contours of given meshes
    /// Generated from function `MR::findIntersectionContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> FindIntersectionContours(MR.Const_Mesh meshA, MR.Const_Mesh meshB, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findIntersectionContours", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_findIntersectionContours(MR.Const_Mesh._Underlying *meshA, MR.Const_Mesh._Underlying *meshB, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_findIntersectionContours(meshA._UnderlyingPtr, meshB._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /** \brief Returns the points of mesh boolean's result mesh
    *
    *
    * Returns vertices and intersection points of mesh that is result of boolean operation of mesh `A` and mesh `B`.
    * Can be used as fast alternative for cases where the mesh topology can be ignored (bounding box, convex hull, etc.)
    * \param meshA Input mesh `A`
    * \param meshB Input mesh `B`
    * \param operation Boolean operation to perform
    * \param rigidB2A Transform from mesh `B` space to mesh `A` space
    */
    /// Generated from function `MR::getBooleanPoints`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRBooleanResultPoints_StdString> GetBooleanPoints(MR.Const_Mesh meshA, MR.Const_Mesh meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getBooleanPoints", ExactSpelling = true)]
        extern static MR.Expected_MRBooleanResultPoints_StdString._Underlying *__MR_getBooleanPoints(MR.Const_Mesh._Underlying *meshA, MR.Const_Mesh._Underlying *meshB, MR.BooleanOperation operation, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Expected_MRBooleanResultPoints_StdString(__MR_getBooleanPoints(meshA._UnderlyingPtr, meshB._UnderlyingPtr, operation, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /// converts all vertices of the mesh first in integer-coordinates, and then back in float-coordinates;
    /// this is necessary to avoid small self-intersections near newly introduced vertices on cut-contours;
    /// this actually sligntly moves only small perentage of vertices near x=0 or y=0 or z=0 planes, where float-precision is higher than int-precision;
    /// newly introduced vertices on cut-contours are also converted, but we expected that they remain unchanged due to idempotent property of the conversion
    /// Generated from function `MR::convertIntFloatAllVerts`.
    public static unsafe void ConvertIntFloatAllVerts(MR.Mesh mesh, MR.Const_CoordinateConverters conv)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertIntFloatAllVerts", ExactSpelling = true)]
        extern static void __MR_convertIntFloatAllVerts(MR.Mesh._Underlying *mesh, MR.Const_CoordinateConverters._Underlying *conv);
        __MR_convertIntFloatAllVerts(mesh._UnderlyingPtr, conv._UnderlyingPtr);
    }
}

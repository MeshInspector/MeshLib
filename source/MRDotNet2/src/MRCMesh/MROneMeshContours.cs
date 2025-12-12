public static partial class MR
{
    // Special data to sort intersections more accurate
    /// Generated from class `MR::SortIntersectionsData`.
    /// This is the const half of the class.
    public class Const_SortIntersectionsData : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SortIntersectionsData(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Destroy", ExactSpelling = true)]
            extern static void __MR_SortIntersectionsData_Destroy(_Underlying *_this);
            __MR_SortIntersectionsData_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SortIntersectionsData() {Dispose(false);}

        public unsafe MR.Const_Mesh OtherMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_otherMesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_SortIntersectionsData_Get_otherMesh(_Underlying *_this);
                return new(__MR_SortIntersectionsData_Get_otherMesh(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Vector_StdVectorMRVarEdgeTri Contours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_contours", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *__MR_SortIntersectionsData_Get_contours(_Underlying *_this);
                return new(__MR_SortIntersectionsData_Get_contours(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_MRVector3iFuncFromConstMRVector3fRef Converter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_converter", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *__MR_SortIntersectionsData_Get_converter(_Underlying *_this);
                return new(__MR_SortIntersectionsData_Get_converter(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe ref readonly MR.AffineXf3f * RigidB2A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_rigidB2A", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_SortIntersectionsData_Get_rigidB2A(_Underlying *_this);
                return ref *__MR_SortIntersectionsData_Get_rigidB2A(_UnderlyingPtr);
            }
        }

        public unsafe ulong MeshAVertsNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_meshAVertsNum", ExactSpelling = true)]
                extern static ulong *__MR_SortIntersectionsData_Get_meshAVertsNum(_Underlying *_this);
                return *__MR_SortIntersectionsData_Get_meshAVertsNum(_UnderlyingPtr);
            }
        }

        public unsafe bool IsOtherA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_Get_isOtherA", ExactSpelling = true)]
                extern static bool *__MR_SortIntersectionsData_Get_isOtherA(_Underlying *_this);
                return *__MR_SortIntersectionsData_Get_isOtherA(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::SortIntersectionsData::SortIntersectionsData`.
        public unsafe Const_SortIntersectionsData(MR._ByValue_SortIntersectionsData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SortIntersectionsData._Underlying *__MR_SortIntersectionsData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SortIntersectionsData._Underlying *_other);
            _UnderlyingPtr = __MR_SortIntersectionsData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs `MR::SortIntersectionsData` elementwise.
        public unsafe Const_SortIntersectionsData(MR.Const_Mesh otherMesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri contours, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef converter, MR.Const_AffineXf3f? rigidB2A, ulong meshAVertsNum, bool isOtherA) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_ConstructFrom", ExactSpelling = true)]
            extern static MR.SortIntersectionsData._Underlying *__MR_SortIntersectionsData_ConstructFrom(MR.Const_Mesh._Underlying *otherMesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *contours, MR.Misc._PassBy converter_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *converter, MR.Const_AffineXf3f._Underlying *rigidB2A, ulong meshAVertsNum, byte isOtherA);
            _UnderlyingPtr = __MR_SortIntersectionsData_ConstructFrom(otherMesh._UnderlyingPtr, contours._UnderlyingPtr, converter.PassByMode, converter.Value is not null ? converter.Value._UnderlyingPtr : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, meshAVertsNum, isOtherA ? (byte)1 : (byte)0);
        }
    }

    // Special data to sort intersections more accurate
    /// Generated from class `MR::SortIntersectionsData`.
    /// This is the non-const half of the class.
    public class SortIntersectionsData : Const_SortIntersectionsData
    {
        internal unsafe SortIntersectionsData(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef Converter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_GetMutable_converter", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *__MR_SortIntersectionsData_GetMutable_converter(_Underlying *_this);
                return new(__MR_SortIntersectionsData_GetMutable_converter(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref readonly MR.AffineXf3f * RigidB2A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_GetMutable_rigidB2A", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_SortIntersectionsData_GetMutable_rigidB2A(_Underlying *_this);
                return ref *__MR_SortIntersectionsData_GetMutable_rigidB2A(_UnderlyingPtr);
            }
        }

        public new unsafe ref ulong MeshAVertsNum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_GetMutable_meshAVertsNum", ExactSpelling = true)]
                extern static ulong *__MR_SortIntersectionsData_GetMutable_meshAVertsNum(_Underlying *_this);
                return ref *__MR_SortIntersectionsData_GetMutable_meshAVertsNum(_UnderlyingPtr);
            }
        }

        public new unsafe ref bool IsOtherA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_GetMutable_isOtherA", ExactSpelling = true)]
                extern static bool *__MR_SortIntersectionsData_GetMutable_isOtherA(_Underlying *_this);
                return ref *__MR_SortIntersectionsData_GetMutable_isOtherA(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::SortIntersectionsData::SortIntersectionsData`.
        public unsafe SortIntersectionsData(MR._ByValue_SortIntersectionsData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SortIntersectionsData._Underlying *__MR_SortIntersectionsData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SortIntersectionsData._Underlying *_other);
            _UnderlyingPtr = __MR_SortIntersectionsData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs `MR::SortIntersectionsData` elementwise.
        public unsafe SortIntersectionsData(MR.Const_Mesh otherMesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri contours, MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef converter, MR.Const_AffineXf3f? rigidB2A, ulong meshAVertsNum, bool isOtherA) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SortIntersectionsData_ConstructFrom", ExactSpelling = true)]
            extern static MR.SortIntersectionsData._Underlying *__MR_SortIntersectionsData_ConstructFrom(MR.Const_Mesh._Underlying *otherMesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *contours, MR.Misc._PassBy converter_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *converter, MR.Const_AffineXf3f._Underlying *rigidB2A, ulong meshAVertsNum, byte isOtherA);
            _UnderlyingPtr = __MR_SortIntersectionsData_ConstructFrom(otherMesh._UnderlyingPtr, contours._UnderlyingPtr, converter.PassByMode, converter.Value is not null ? converter.Value._UnderlyingPtr : null, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, meshAVertsNum, isOtherA ? (byte)1 : (byte)0);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SortIntersectionsData` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SortIntersectionsData`/`Const_SortIntersectionsData` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SortIntersectionsData
    {
        internal readonly Const_SortIntersectionsData? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SortIntersectionsData(Const_SortIntersectionsData new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SortIntersectionsData(Const_SortIntersectionsData arg) {return new(arg);}
        public _ByValue_SortIntersectionsData(MR.Misc._Moved<SortIntersectionsData> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SortIntersectionsData(MR.Misc._Moved<SortIntersectionsData> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SortIntersectionsData` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SortIntersectionsData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SortIntersectionsData`/`Const_SortIntersectionsData` directly.
    public class _InOptMut_SortIntersectionsData
    {
        public SortIntersectionsData? Opt;

        public _InOptMut_SortIntersectionsData() {}
        public _InOptMut_SortIntersectionsData(SortIntersectionsData value) {Opt = value;}
        public static implicit operator _InOptMut_SortIntersectionsData(SortIntersectionsData value) {return new(value);}
    }

    /// This is used for optional parameters of class `SortIntersectionsData` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SortIntersectionsData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SortIntersectionsData`/`Const_SortIntersectionsData` to pass it to the function.
    public class _InOptConst_SortIntersectionsData
    {
        public Const_SortIntersectionsData? Opt;

        public _InOptConst_SortIntersectionsData() {}
        public _InOptConst_SortIntersectionsData(Const_SortIntersectionsData value) {Opt = value;}
        public static implicit operator _InOptConst_SortIntersectionsData(Const_SortIntersectionsData value) {return new(value);}
    }

    // Simple point on mesh, represented by primitive id and coordinate in mesh space
    /// Generated from class `MR::OneMeshIntersection`.
    /// This is the const half of the class.
    public class Const_OneMeshIntersection : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OneMeshIntersection(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_Destroy", ExactSpelling = true)]
            extern static void __MR_OneMeshIntersection_Destroy(_Underlying *_this);
            __MR_OneMeshIntersection_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OneMeshIntersection() {Dispose(false);}

        public unsafe MR.Std.Const_Variant_MRFaceId_MREdgeId_MRVertId PrimitiveId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_Get_primitiveId", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRFaceId_MREdgeId_MRVertId._Underlying *__MR_OneMeshIntersection_Get_primitiveId(_Underlying *_this);
                return new(__MR_OneMeshIntersection_Get_primitiveId(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f Coordinate
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_Get_coordinate", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_OneMeshIntersection_Get_coordinate(_Underlying *_this);
                return new(__MR_OneMeshIntersection_Get_coordinate(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OneMeshIntersection() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_DefaultConstruct();
            _UnderlyingPtr = __MR_OneMeshIntersection_DefaultConstruct();
        }

        /// Constructs `MR::OneMeshIntersection` elementwise.
        public unsafe Const_OneMeshIntersection(MR.Std.Const_Variant_MRFaceId_MREdgeId_MRVertId primitiveId, MR.Vector3f coordinate) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_ConstructFrom", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_ConstructFrom(MR.Std.Variant_MRFaceId_MREdgeId_MRVertId._Underlying *primitiveId, MR.Vector3f coordinate);
            _UnderlyingPtr = __MR_OneMeshIntersection_ConstructFrom(primitiveId._UnderlyingPtr, coordinate);
        }

        /// Generated from constructor `MR::OneMeshIntersection::OneMeshIntersection`.
        public unsafe Const_OneMeshIntersection(MR.Const_OneMeshIntersection _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_ConstructFromAnother(MR.OneMeshIntersection._Underlying *_other);
            _UnderlyingPtr = __MR_OneMeshIntersection_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // Simple point on mesh, represented by primitive id and coordinate in mesh space
    /// Generated from class `MR::OneMeshIntersection`.
    /// This is the non-const half of the class.
    public class OneMeshIntersection : Const_OneMeshIntersection
    {
        internal unsafe OneMeshIntersection(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Variant_MRFaceId_MREdgeId_MRVertId PrimitiveId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_GetMutable_primitiveId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceId_MREdgeId_MRVertId._Underlying *__MR_OneMeshIntersection_GetMutable_primitiveId(_Underlying *_this);
                return new(__MR_OneMeshIntersection_GetMutable_primitiveId(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f Coordinate
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_GetMutable_coordinate", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_OneMeshIntersection_GetMutable_coordinate(_Underlying *_this);
                return new(__MR_OneMeshIntersection_GetMutable_coordinate(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OneMeshIntersection() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_DefaultConstruct();
            _UnderlyingPtr = __MR_OneMeshIntersection_DefaultConstruct();
        }

        /// Constructs `MR::OneMeshIntersection` elementwise.
        public unsafe OneMeshIntersection(MR.Std.Const_Variant_MRFaceId_MREdgeId_MRVertId primitiveId, MR.Vector3f coordinate) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_ConstructFrom", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_ConstructFrom(MR.Std.Variant_MRFaceId_MREdgeId_MRVertId._Underlying *primitiveId, MR.Vector3f coordinate);
            _UnderlyingPtr = __MR_OneMeshIntersection_ConstructFrom(primitiveId._UnderlyingPtr, coordinate);
        }

        /// Generated from constructor `MR::OneMeshIntersection::OneMeshIntersection`.
        public unsafe OneMeshIntersection(MR.Const_OneMeshIntersection _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_ConstructFromAnother(MR.OneMeshIntersection._Underlying *_other);
            _UnderlyingPtr = __MR_OneMeshIntersection_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OneMeshIntersection::operator=`.
        public unsafe MR.OneMeshIntersection Assign(MR.Const_OneMeshIntersection _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshIntersection_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshIntersection._Underlying *__MR_OneMeshIntersection_AssignFromAnother(_Underlying *_this, MR.OneMeshIntersection._Underlying *_other);
            return new(__MR_OneMeshIntersection_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OneMeshIntersection` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OneMeshIntersection`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OneMeshIntersection`/`Const_OneMeshIntersection` directly.
    public class _InOptMut_OneMeshIntersection
    {
        public OneMeshIntersection? Opt;

        public _InOptMut_OneMeshIntersection() {}
        public _InOptMut_OneMeshIntersection(OneMeshIntersection value) {Opt = value;}
        public static implicit operator _InOptMut_OneMeshIntersection(OneMeshIntersection value) {return new(value);}
    }

    /// This is used for optional parameters of class `OneMeshIntersection` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OneMeshIntersection`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OneMeshIntersection`/`Const_OneMeshIntersection` to pass it to the function.
    public class _InOptConst_OneMeshIntersection
    {
        public Const_OneMeshIntersection? Opt;

        public _InOptConst_OneMeshIntersection() {}
        public _InOptConst_OneMeshIntersection(Const_OneMeshIntersection value) {Opt = value;}
        public static implicit operator _InOptConst_OneMeshIntersection(Const_OneMeshIntersection value) {return new(value);}
    }

    // One contour on mesh
    /// Generated from class `MR::OneMeshContour`.
    /// This is the const half of the class.
    public class Const_OneMeshContour : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OneMeshContour(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_Destroy", ExactSpelling = true)]
            extern static void __MR_OneMeshContour_Destroy(_Underlying *_this);
            __MR_OneMeshContour_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OneMeshContour() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_MROneMeshIntersection Intersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_Get_intersections", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MROneMeshIntersection._Underlying *__MR_OneMeshContour_Get_intersections(_Underlying *_this);
                return new(__MR_OneMeshContour_Get_intersections(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe bool Closed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_Get_closed", ExactSpelling = true)]
                extern static bool *__MR_OneMeshContour_Get_closed(_Underlying *_this);
                return *__MR_OneMeshContour_Get_closed(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OneMeshContour() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_DefaultConstruct();
            _UnderlyingPtr = __MR_OneMeshContour_DefaultConstruct();
        }

        /// Constructs `MR::OneMeshContour` elementwise.
        public unsafe Const_OneMeshContour(MR.Std._ByValue_Vector_MROneMeshIntersection intersections, bool closed) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_ConstructFrom", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_ConstructFrom(MR.Misc._PassBy intersections_pass_by, MR.Std.Vector_MROneMeshIntersection._Underlying *intersections, byte closed);
            _UnderlyingPtr = __MR_OneMeshContour_ConstructFrom(intersections.PassByMode, intersections.Value is not null ? intersections.Value._UnderlyingPtr : null, closed ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::OneMeshContour::OneMeshContour`.
        public unsafe Const_OneMeshContour(MR._ByValue_OneMeshContour _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OneMeshContour._Underlying *_other);
            _UnderlyingPtr = __MR_OneMeshContour_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    // One contour on mesh
    /// Generated from class `MR::OneMeshContour`.
    /// This is the non-const half of the class.
    public class OneMeshContour : Const_OneMeshContour
    {
        internal unsafe OneMeshContour(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_MROneMeshIntersection Intersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_GetMutable_intersections", ExactSpelling = true)]
                extern static MR.Std.Vector_MROneMeshIntersection._Underlying *__MR_OneMeshContour_GetMutable_intersections(_Underlying *_this);
                return new(__MR_OneMeshContour_GetMutable_intersections(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref bool Closed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_GetMutable_closed", ExactSpelling = true)]
                extern static bool *__MR_OneMeshContour_GetMutable_closed(_Underlying *_this);
                return ref *__MR_OneMeshContour_GetMutable_closed(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OneMeshContour() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_DefaultConstruct();
            _UnderlyingPtr = __MR_OneMeshContour_DefaultConstruct();
        }

        /// Constructs `MR::OneMeshContour` elementwise.
        public unsafe OneMeshContour(MR.Std._ByValue_Vector_MROneMeshIntersection intersections, bool closed) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_ConstructFrom", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_ConstructFrom(MR.Misc._PassBy intersections_pass_by, MR.Std.Vector_MROneMeshIntersection._Underlying *intersections, byte closed);
            _UnderlyingPtr = __MR_OneMeshContour_ConstructFrom(intersections.PassByMode, intersections.Value is not null ? intersections.Value._UnderlyingPtr : null, closed ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::OneMeshContour::OneMeshContour`.
        public unsafe OneMeshContour(MR._ByValue_OneMeshContour _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OneMeshContour._Underlying *_other);
            _UnderlyingPtr = __MR_OneMeshContour_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::OneMeshContour::operator=`.
        public unsafe MR.OneMeshContour Assign(MR._ByValue_OneMeshContour _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OneMeshContour_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OneMeshContour._Underlying *__MR_OneMeshContour_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.OneMeshContour._Underlying *_other);
            return new(__MR_OneMeshContour_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `OneMeshContour` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `OneMeshContour`/`Const_OneMeshContour` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_OneMeshContour
    {
        internal readonly Const_OneMeshContour? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_OneMeshContour() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_OneMeshContour(Const_OneMeshContour new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_OneMeshContour(Const_OneMeshContour arg) {return new(arg);}
        public _ByValue_OneMeshContour(MR.Misc._Moved<OneMeshContour> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_OneMeshContour(MR.Misc._Moved<OneMeshContour> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `OneMeshContour` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OneMeshContour`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OneMeshContour`/`Const_OneMeshContour` directly.
    public class _InOptMut_OneMeshContour
    {
        public OneMeshContour? Opt;

        public _InOptMut_OneMeshContour() {}
        public _InOptMut_OneMeshContour(OneMeshContour value) {Opt = value;}
        public static implicit operator _InOptMut_OneMeshContour(OneMeshContour value) {return new(value);}
    }

    /// This is used for optional parameters of class `OneMeshContour` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OneMeshContour`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OneMeshContour`/`Const_OneMeshContour` to pass it to the function.
    public class _InOptConst_OneMeshContour
    {
        public Const_OneMeshContour? Opt;

        public _InOptConst_OneMeshContour() {}
        public _InOptConst_OneMeshContour(Const_OneMeshContour value) {Opt = value;}
        public static implicit operator _InOptConst_OneMeshContour(Const_OneMeshContour value) {return new(value);}
    }

    /// Geo path search settings
    /// Generated from class `MR::SearchPathSettings`.
    /// This is the const half of the class.
    public class Const_SearchPathSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SearchPathSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SearchPathSettings_Destroy(_Underlying *_this);
            __MR_SearchPathSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SearchPathSettings() {Dispose(false);}

        ///< the algorithm to compute approximately geodesic path
        public unsafe MR.GeodesicPathApprox GeodesicPathApprox
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_Get_geodesicPathApprox", ExactSpelling = true)]
                extern static MR.GeodesicPathApprox *__MR_SearchPathSettings_Get_geodesicPathApprox(_Underlying *_this);
                return *__MR_SearchPathSettings_Get_geodesicPathApprox(_UnderlyingPtr);
            }
        }

        ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
        public unsafe int MaxReduceIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_Get_maxReduceIters", ExactSpelling = true)]
                extern static int *__MR_SearchPathSettings_Get_maxReduceIters(_Underlying *_this);
                return *__MR_SearchPathSettings_Get_maxReduceIters(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SearchPathSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SearchPathSettings_DefaultConstruct();
        }

        /// Constructs `MR::SearchPathSettings` elementwise.
        public unsafe Const_SearchPathSettings(MR.GeodesicPathApprox geodesicPathApprox, int maxReduceIters) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_ConstructFrom(MR.GeodesicPathApprox geodesicPathApprox, int maxReduceIters);
            _UnderlyingPtr = __MR_SearchPathSettings_ConstructFrom(geodesicPathApprox, maxReduceIters);
        }

        /// Generated from constructor `MR::SearchPathSettings::SearchPathSettings`.
        public unsafe Const_SearchPathSettings(MR.Const_SearchPathSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_ConstructFromAnother(MR.SearchPathSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SearchPathSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Geo path search settings
    /// Generated from class `MR::SearchPathSettings`.
    /// This is the non-const half of the class.
    public class SearchPathSettings : Const_SearchPathSettings
    {
        internal unsafe SearchPathSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< the algorithm to compute approximately geodesic path
        public new unsafe ref MR.GeodesicPathApprox GeodesicPathApprox
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_GetMutable_geodesicPathApprox", ExactSpelling = true)]
                extern static MR.GeodesicPathApprox *__MR_SearchPathSettings_GetMutable_geodesicPathApprox(_Underlying *_this);
                return ref *__MR_SearchPathSettings_GetMutable_geodesicPathApprox(_UnderlyingPtr);
            }
        }

        ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
        public new unsafe ref int MaxReduceIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_GetMutable_maxReduceIters", ExactSpelling = true)]
                extern static int *__MR_SearchPathSettings_GetMutable_maxReduceIters(_Underlying *_this);
                return ref *__MR_SearchPathSettings_GetMutable_maxReduceIters(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SearchPathSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SearchPathSettings_DefaultConstruct();
        }

        /// Constructs `MR::SearchPathSettings` elementwise.
        public unsafe SearchPathSettings(MR.GeodesicPathApprox geodesicPathApprox, int maxReduceIters) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_ConstructFrom(MR.GeodesicPathApprox geodesicPathApprox, int maxReduceIters);
            _UnderlyingPtr = __MR_SearchPathSettings_ConstructFrom(geodesicPathApprox, maxReduceIters);
        }

        /// Generated from constructor `MR::SearchPathSettings::SearchPathSettings`.
        public unsafe SearchPathSettings(MR.Const_SearchPathSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_ConstructFromAnother(MR.SearchPathSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SearchPathSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SearchPathSettings::operator=`.
        public unsafe MR.SearchPathSettings Assign(MR.Const_SearchPathSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SearchPathSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SearchPathSettings._Underlying *__MR_SearchPathSettings_AssignFromAnother(_Underlying *_this, MR.SearchPathSettings._Underlying *_other);
            return new(__MR_SearchPathSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SearchPathSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SearchPathSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SearchPathSettings`/`Const_SearchPathSettings` directly.
    public class _InOptMut_SearchPathSettings
    {
        public SearchPathSettings? Opt;

        public _InOptMut_SearchPathSettings() {}
        public _InOptMut_SearchPathSettings(SearchPathSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SearchPathSettings(SearchPathSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SearchPathSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SearchPathSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SearchPathSettings`/`Const_SearchPathSettings` to pass it to the function.
    public class _InOptConst_SearchPathSettings
    {
        public Const_SearchPathSettings? Opt;

        public _InOptConst_SearchPathSettings() {}
        public _InOptConst_SearchPathSettings(Const_SearchPathSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SearchPathSettings(Const_SearchPathSettings value) {return new(value);}
    }

    // Divides faces that fully own contours into 3 parts with center in center mass of one of the face contours
    // if there is more than one contour on face it guarantee to subdivide at least one lone contour on this face
    /// Generated from function `MR::subdivideLoneContours`.
    public static unsafe void SubdivideLoneContours(MR.Mesh mesh, MR.Std.Const_Vector_MROneMeshContour contours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2oldMap = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdivideLoneContours", ExactSpelling = true)]
        extern static void __MR_subdivideLoneContours(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MROneMeshContour._Underlying *contours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2oldMap);
        __MR_subdivideLoneContours(mesh._UnderlyingPtr, contours._UnderlyingPtr, new2oldMap is not null ? new2oldMap._UnderlyingPtr : null);
    }

    /// Converts contours given in topological terms as the intersections of one mesh's edge and another mesh's triangle (ContinuousContours),
    /// into contours of meshA and/or meshB given as a sequence of (primitiveId and Cartesian coordinates);
    /// converters are required for better precision in case of degenerations;
    /// note that contours should not have intersections
    /// Generated from function `MR::getOneMeshIntersectionContours`.
    /// Parameter `addSelfyTerminalVerts` defaults to `false`.
    public static unsafe void GetOneMeshIntersectionContours(MR.Const_Mesh meshA, MR.Const_Mesh meshB, MR.Std.Const_Vector_StdVectorMRVarEdgeTri contours, MR.Std.Vector_MROneMeshContour? outA, MR.Std.Vector_MROneMeshContour? outB, MR.Const_CoordinateConverters converters, MR.Const_AffineXf3f? rigidB2A = null, MR.Std.Vector_StdVectorMRVector3f? outPtsA = null, bool? addSelfyTerminalVerts = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getOneMeshIntersectionContours", ExactSpelling = true)]
        extern static void __MR_getOneMeshIntersectionContours(MR.Const_Mesh._Underlying *meshA, MR.Const_Mesh._Underlying *meshB, MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *contours, MR.Std.Vector_MROneMeshContour._Underlying *outA, MR.Std.Vector_MROneMeshContour._Underlying *outB, MR.Const_CoordinateConverters._Underlying *converters, MR.Const_AffineXf3f._Underlying *rigidB2A, MR.Std.Vector_StdVectorMRVector3f._Underlying *outPtsA, byte *addSelfyTerminalVerts);
        byte __deref_addSelfyTerminalVerts = addSelfyTerminalVerts.GetValueOrDefault() ? (byte)1 : (byte)0;
        __MR_getOneMeshIntersectionContours(meshA._UnderlyingPtr, meshB._UnderlyingPtr, contours._UnderlyingPtr, outA is not null ? outA._UnderlyingPtr : null, outB is not null ? outB._UnderlyingPtr : null, converters._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, outPtsA is not null ? outPtsA._UnderlyingPtr : null, addSelfyTerminalVerts.HasValue ? &__deref_addSelfyTerminalVerts : null);
    }

    // Converts ordered continuous self contours of single meshes to OneMeshContours
    // converters are required for better precision in case of degenerations
    /// Generated from function `MR::getOneMeshSelfIntersectionContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MROneMeshContour> GetOneMeshSelfIntersectionContours(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri contours, MR.Const_CoordinateConverters converters, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getOneMeshSelfIntersectionContours", ExactSpelling = true)]
        extern static MR.Std.Vector_MROneMeshContour._Underlying *__MR_getOneMeshSelfIntersectionContours(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *contours, MR.Const_CoordinateConverters._Underlying *converters, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Vector_MROneMeshContour(__MR_getOneMeshSelfIntersectionContours(mesh._UnderlyingPtr, contours._UnderlyingPtr, converters._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    // Converts OneMeshContours contours representation to Contours3f: set of coordinates
    /// Generated from function `MR::extractMeshContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> ExtractMeshContours(MR.Std.Const_Vector_MROneMeshContour meshContours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractMeshContours", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_extractMeshContours(MR.Std.Const_Vector_MROneMeshContour._Underlying *meshContours);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_extractMeshContours(meshContours._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief Makes continuous contour by mesh tri points, if first and last meshTriPoint is the same, makes closed contour
    *
    * Finds shortest paths between neighbor \p surfaceLine and build contour MR::cutMesh input
    * \param searchSettings settings for search geo path 
    * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
    */
    /// Generated from function `MR::convertMeshTriPointsToMeshContour`.
    /// Parameter `searchSettings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MROneMeshContour_StdString> ConvertMeshTriPointsToMeshContour(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, MR.Const_SearchPathSettings? searchSettings = null, MR.Std.Vector_Int? pivotIndices = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertMeshTriPointsToMeshContour", ExactSpelling = true)]
        extern static MR.Expected_MROneMeshContour_StdString._Underlying *__MR_convertMeshTriPointsToMeshContour(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, MR.SearchPathSettings._Underlying *searchSettings, MR.Std.Vector_Int._Underlying *pivotIndices);
        return MR.Misc.Move(new MR.Expected_MROneMeshContour_StdString(__MR_convertMeshTriPointsToMeshContour(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, searchSettings is not null ? searchSettings._UnderlyingPtr : null, pivotIndices is not null ? pivotIndices._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Makes closed continuous contour by mesh tri points, note that first and last meshTriPoint should not be same
    * 
    * Finds shortest paths between neighbor \p surfaceLine and build closed contour MR::cutMesh input
    * \param pivotIndices optional output indices of given surfaceLine in result OneMeshContour
    * \note better use convertMeshTriPointsToMeshContour(...) instead, note that it requires same front and back MeshTriPoints for closed contour
    */
    /// Generated from function `MR::convertMeshTriPointsToClosedContour`.
    /// Parameter `searchSettings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MROneMeshContour_StdString> ConvertMeshTriPointsToClosedContour(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, MR.Const_SearchPathSettings? searchSettings = null, MR.Std.Vector_Int? pivotIndices = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertMeshTriPointsToClosedContour", ExactSpelling = true)]
        extern static MR.Expected_MROneMeshContour_StdString._Underlying *__MR_convertMeshTriPointsToClosedContour(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, MR.SearchPathSettings._Underlying *searchSettings, MR.Std.Vector_Int._Underlying *pivotIndices);
        return MR.Misc.Move(new MR.Expected_MROneMeshContour_StdString(__MR_convertMeshTriPointsToClosedContour(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, searchSettings is not null ? searchSettings._UnderlyingPtr : null, pivotIndices is not null ? pivotIndices._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Converts SurfacePath to OneMeshContours
    *
    * Creates MR::OneMeshContour object from given surface path with ends for MR::cutMesh input
    * `start` and surfacePath.front() should be from same face
    * surfacePath.back() and `end` should be from same face
    * 
    * note that whole path (including `start` and `end`) should not have self-intersections
    * also following case is not supported (vertex -> edge (incident with vertex)):
    * 
    * vert path  edge point path     edge end
    * o----------o-  --  --  --  --  O
    *  \          \                /
    *       \      \          /
    *            \  \     /
    *               \\/
    *                 o path
    */
    /// Generated from function `MR::convertSurfacePathWithEndsToMeshContour`.
    public static unsafe MR.Misc._Moved<MR.OneMeshContour> ConvertSurfacePathWithEndsToMeshContour(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Std.Const_Vector_MREdgePoint surfacePath, MR.Const_MeshTriPoint end)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertSurfacePathWithEndsToMeshContour", ExactSpelling = true)]
        extern static MR.OneMeshContour._Underlying *__MR_convertSurfacePathWithEndsToMeshContour(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Std.Const_Vector_MREdgePoint._Underlying *surfacePath, MR.Const_MeshTriPoint._Underlying *end);
        return MR.Misc.Move(new MR.OneMeshContour(__MR_convertSurfacePathWithEndsToMeshContour(mesh._UnderlyingPtr, start._UnderlyingPtr, surfacePath._UnderlyingPtr, end._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief Converts SurfacePaths to OneMeshContours
    * 
    * Creates MR::OneMeshContours object from given surface paths for MR::cutMesh input
    */
    /// Generated from function `MR::convertSurfacePathsToMeshContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MROneMeshContour> ConvertSurfacePathsToMeshContours(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgePoint surfacePaths)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertSurfacePathsToMeshContours", ExactSpelling = true)]
        extern static MR.Std.Vector_MROneMeshContour._Underlying *__MR_convertSurfacePathsToMeshContours(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgePoint._Underlying *surfacePaths);
        return MR.Misc.Move(new MR.Std.Vector_MROneMeshContour(__MR_convertSurfacePathsToMeshContours(mesh._UnderlyingPtr, surfacePaths._UnderlyingPtr), is_owning: true));
    }
}

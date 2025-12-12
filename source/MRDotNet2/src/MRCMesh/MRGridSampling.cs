public static partial class MR
{
    /// structure to contain pointers to model data
    /// Generated from class `MR::ModelPointsData`.
    /// This is the const half of the class.
    public class Const_ModelPointsData : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ModelPointsData(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_Destroy", ExactSpelling = true)]
            extern static void __MR_ModelPointsData_Destroy(_Underlying *_this);
            __MR_ModelPointsData_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ModelPointsData() {Dispose(false);}

        /// all points of model
        public unsafe ref readonly void * Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_Get_points", ExactSpelling = true)]
                extern static void **__MR_ModelPointsData_Get_points(_Underlying *_this);
                return ref *__MR_ModelPointsData_Get_points(_UnderlyingPtr);
            }
        }

        /// bitset of valid points
        public unsafe ref readonly void * ValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_Get_validPoints", ExactSpelling = true)]
                extern static void **__MR_ModelPointsData_Get_validPoints(_Underlying *_this);
                return ref *__MR_ModelPointsData_Get_validPoints(_UnderlyingPtr);
            }
        }

        /// model world xf
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ModelPointsData_Get_xf(_Underlying *_this);
                return ref *__MR_ModelPointsData_Get_xf(_UnderlyingPtr);
            }
        }

        /// if present this value will override ObjId in result ObjVertId
        public unsafe MR.Const_ObjId FakeObjId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_Get_fakeObjId", ExactSpelling = true)]
                extern static MR.Const_ObjId._Underlying *__MR_ModelPointsData_Get_fakeObjId(_Underlying *_this);
                return new(__MR_ModelPointsData_Get_fakeObjId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ModelPointsData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_DefaultConstruct();
            _UnderlyingPtr = __MR_ModelPointsData_DefaultConstruct();
        }

        /// Constructs `MR::ModelPointsData` elementwise.
        public unsafe Const_ModelPointsData(MR.Const_VertCoords? points, MR.Const_VertBitSet? validPoints, MR.Const_AffineXf3f? xf, MR.ObjId fakeObjId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_ConstructFrom(MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validPoints, MR.Const_AffineXf3f._Underlying *xf, MR.ObjId fakeObjId);
            _UnderlyingPtr = __MR_ModelPointsData_ConstructFrom(points is not null ? points._UnderlyingPtr : null, validPoints is not null ? validPoints._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null, fakeObjId);
        }

        /// Generated from constructor `MR::ModelPointsData::ModelPointsData`.
        public unsafe Const_ModelPointsData(MR.Const_ModelPointsData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_ConstructFromAnother(MR.ModelPointsData._Underlying *_other);
            _UnderlyingPtr = __MR_ModelPointsData_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// structure to contain pointers to model data
    /// Generated from class `MR::ModelPointsData`.
    /// This is the non-const half of the class.
    public class ModelPointsData : Const_ModelPointsData
    {
        internal unsafe ModelPointsData(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// all points of model
        public new unsafe ref readonly void * Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_GetMutable_points", ExactSpelling = true)]
                extern static void **__MR_ModelPointsData_GetMutable_points(_Underlying *_this);
                return ref *__MR_ModelPointsData_GetMutable_points(_UnderlyingPtr);
            }
        }

        /// bitset of valid points
        public new unsafe ref readonly void * ValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_GetMutable_validPoints", ExactSpelling = true)]
                extern static void **__MR_ModelPointsData_GetMutable_validPoints(_Underlying *_this);
                return ref *__MR_ModelPointsData_GetMutable_validPoints(_UnderlyingPtr);
            }
        }

        /// model world xf
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ModelPointsData_GetMutable_xf(_Underlying *_this);
                return ref *__MR_ModelPointsData_GetMutable_xf(_UnderlyingPtr);
            }
        }

        /// if present this value will override ObjId in result ObjVertId
        public new unsafe MR.Mut_ObjId FakeObjId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_GetMutable_fakeObjId", ExactSpelling = true)]
                extern static MR.Mut_ObjId._Underlying *__MR_ModelPointsData_GetMutable_fakeObjId(_Underlying *_this);
                return new(__MR_ModelPointsData_GetMutable_fakeObjId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ModelPointsData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_DefaultConstruct();
            _UnderlyingPtr = __MR_ModelPointsData_DefaultConstruct();
        }

        /// Constructs `MR::ModelPointsData` elementwise.
        public unsafe ModelPointsData(MR.Const_VertCoords? points, MR.Const_VertBitSet? validPoints, MR.Const_AffineXf3f? xf, MR.ObjId fakeObjId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_ConstructFrom(MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validPoints, MR.Const_AffineXf3f._Underlying *xf, MR.ObjId fakeObjId);
            _UnderlyingPtr = __MR_ModelPointsData_ConstructFrom(points is not null ? points._UnderlyingPtr : null, validPoints is not null ? validPoints._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null, fakeObjId);
        }

        /// Generated from constructor `MR::ModelPointsData::ModelPointsData`.
        public unsafe ModelPointsData(MR.Const_ModelPointsData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_ConstructFromAnother(MR.ModelPointsData._Underlying *_other);
            _UnderlyingPtr = __MR_ModelPointsData_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ModelPointsData::operator=`.
        public unsafe MR.ModelPointsData Assign(MR.Const_ModelPointsData _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelPointsData_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ModelPointsData._Underlying *__MR_ModelPointsData_AssignFromAnother(_Underlying *_this, MR.ModelPointsData._Underlying *_other);
            return new(__MR_ModelPointsData_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ModelPointsData` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ModelPointsData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelPointsData`/`Const_ModelPointsData` directly.
    public class _InOptMut_ModelPointsData
    {
        public ModelPointsData? Opt;

        public _InOptMut_ModelPointsData() {}
        public _InOptMut_ModelPointsData(ModelPointsData value) {Opt = value;}
        public static implicit operator _InOptMut_ModelPointsData(ModelPointsData value) {return new(value);}
    }

    /// This is used for optional parameters of class `ModelPointsData` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ModelPointsData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelPointsData`/`Const_ModelPointsData` to pass it to the function.
    public class _InOptConst_ModelPointsData
    {
        public Const_ModelPointsData? Opt;

        public _InOptConst_ModelPointsData() {}
        public _InOptConst_ModelPointsData(Const_ModelPointsData value) {Opt = value;}
        public static implicit operator _InOptConst_ModelPointsData(Const_ModelPointsData value) {return new(value);}
    }

    /// Generated from class `MR::ObjVertId`.
    /// This is the const reference to the struct.
    public class Const_ObjVertId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_ObjVertId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly ObjVertId UnderlyingStruct => ref *(ObjVertId *)_UnderlyingPtr;

        internal unsafe Const_ObjVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjVertId_Destroy(_Underlying *_this);
            __MR_ObjVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjVertId() {Dispose(false);}

        public ref readonly MR.ObjId ObjId => ref UnderlyingStruct.ObjId;

        public ref readonly MR.VertId VId => ref UnderlyingStruct.VId;

        /// Generated copy constructor.
        public unsafe Const_ObjVertId(Const_ObjVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjVertId __MR_ObjVertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.ObjVertId _ctor_result = __MR_ObjVertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_ObjVertId _1, MR.Const_ObjVertId _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ObjVertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ObjVertId(MR.Const_ObjVertId._Underlying *_1, MR.Const_ObjVertId._Underlying *_2);
            return __MR_equal_MR_ObjVertId(_1._UnderlyingPtr, _2._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ObjVertId _1, MR.Const_ObjVertId _2)
        {
            return !(_1 == _2);
        }

        // IEquatable:

        public bool Equals(MR.Const_ObjVertId? _2)
        {
            if (_2 is null)
                return false;
            return this == _2;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_ObjVertId)
                return this == (MR.Const_ObjVertId)other;
            return false;
        }
    }

    /// Generated from class `MR::ObjVertId`.
    /// This is the non-const reference to the struct.
    public class Mut_ObjVertId : Const_ObjVertId
    {
        /// Get the underlying struct.
        public unsafe new ref ObjVertId UnderlyingStruct => ref *(ObjVertId *)_UnderlyingPtr;

        internal unsafe Mut_ObjVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.ObjId ObjId => ref UnderlyingStruct.ObjId;

        public new ref MR.VertId VId => ref UnderlyingStruct.VId;

        /// Generated copy constructor.
        public unsafe Mut_ObjVertId(Const_ObjVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_ObjVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjVertId __MR_ObjVertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.ObjVertId _ctor_result = __MR_ObjVertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }
    }

    /// Generated from class `MR::ObjVertId`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct ObjVertId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator ObjVertId(Const_ObjVertId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_ObjVertId(ObjVertId other) => new(new Mut_ObjVertId((Mut_ObjVertId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.ObjId ObjId;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public MR.VertId VId;

        /// Generated copy constructor.
        public ObjVertId(ObjVertId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjVertId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjVertId __MR_ObjVertId_DefaultConstruct();
            this = __MR_ObjVertId_DefaultConstruct();
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.ObjVertId _1, MR.ObjVertId _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ObjVertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ObjVertId(MR.Const_ObjVertId._Underlying *_1, MR.Const_ObjVertId._Underlying *_2);
            return __MR_equal_MR_ObjVertId((MR.Mut_ObjVertId._Underlying *)&_1, (MR.Mut_ObjVertId._Underlying *)&_2) != 0;
        }

        public static unsafe bool operator!=(MR.ObjVertId _1, MR.ObjVertId _2)
        {
            return !(_1 == _2);
        }

        // IEquatable:

        public bool Equals(MR.ObjVertId _2)
        {
            return this == _2;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ObjVertId)
                return this == (MR.ObjVertId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_ObjVertId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_ObjVertId`/`Const_ObjVertId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_ObjVertId
    {
        public readonly bool HasValue;
        internal readonly ObjVertId Object;
        public ObjVertId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_ObjVertId() {HasValue = false;}
        public _InOpt_ObjVertId(ObjVertId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_ObjVertId(ObjVertId new_value) {return new(new_value);}
        public _InOpt_ObjVertId(Const_ObjVertId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_ObjVertId(Const_ObjVertId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_ObjVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ObjVertId`/`Const_ObjVertId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `ObjVertId`.
    public class _InOptMut_ObjVertId
    {
        public Mut_ObjVertId? Opt;

        public _InOptMut_ObjVertId() {}
        public _InOptMut_ObjVertId(Mut_ObjVertId value) {Opt = value;}
        public static implicit operator _InOptMut_ObjVertId(Mut_ObjVertId value) {return new(value);}
        public unsafe _InOptMut_ObjVertId(ref ObjVertId value)
        {
            fixed (ObjVertId *value_ptr = &value)
            {
                Opt = new((Const_ObjVertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_ObjVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ObjVertId`/`Const_ObjVertId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `ObjVertId`.
    public class _InOptConst_ObjVertId
    {
        public Const_ObjVertId? Opt;

        public _InOptConst_ObjVertId() {}
        public _InOptConst_ObjVertId(Const_ObjVertId value) {Opt = value;}
        public static implicit operator _InOptConst_ObjVertId(Const_ObjVertId value) {return new(value);}
        public unsafe _InOptConst_ObjVertId(ref readonly ObjVertId value)
        {
            fixed (ObjVertId *value_ptr = &value)
            {
                Opt = new((Const_ObjVertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// performs sampling of mesh vertices;
    /// subdivides mesh bounding box on voxels of approximately given size and returns at most one vertex per voxel;
    /// if voxelSize<=0 then returns all region vertices as samples;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::verticesGridSampling`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> VerticesGridSampling(MR.Const_MeshPart mp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_verticesGridSampling", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_verticesGridSampling(MR.Const_MeshPart._Underlying *mp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_verticesGridSampling(mp._UnderlyingPtr, voxelSize, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// performs sampling of cloud points;
    /// subdivides point cloud bounding box on voxels of approximately given size and returns at most one point per voxel;
    /// if voxelSize<=0 then returns all valid points as samples;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::pointGridSampling`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> PointGridSampling(MR.Const_PointCloudPart pcp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointGridSampling", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_pointGridSampling(MR.Const_PointCloudPart._Underlying *pcp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_pointGridSampling(pcp._UnderlyingPtr, voxelSize, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// performs sampling of several models respecting their world transformations
    /// subdivides models bounding box on voxels of approximately given size and returns at most one point per voxel;
    /// if voxelSize<=0 then returns all points from all models as samples;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::multiModelGridSampling`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_StdVectorMRObjVertId> MultiModelGridSampling(MR.Const_Vector_MRModelPointsData_MRObjId models, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_multiModelGridSampling", ExactSpelling = true)]
        extern static MR.Std.Optional_StdVectorMRObjVertId._Underlying *__MR_multiModelGridSampling(MR.Const_Vector_MRModelPointsData_MRObjId._Underlying *models, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_StdVectorMRObjVertId(__MR_multiModelGridSampling(models._UnderlyingPtr, voxelSize, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }
}

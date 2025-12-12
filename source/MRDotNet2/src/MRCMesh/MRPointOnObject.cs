public static partial class MR
{
    // point located on either
    // 1. face of ObjectMesh
    // 2. line of ObjectLines
    // 3. point of ObjectPoints
    /// Generated from class `MR::PointOnObject`.
    /// This is the const half of the class.
    public class Const_PointOnObject : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointOnObject(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_Destroy", ExactSpelling = true)]
            extern static void __MR_PointOnObject_Destroy(_Underlying *_this);
            __MR_PointOnObject_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointOnObject() {Dispose(false);}

        /// 3D location on the object in local coordinates
        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointOnObject_Get_point(_Underlying *_this);
                return new(__MR_PointOnObject_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// z buffer value
        public unsafe float ZBuffer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_Get_zBuffer", ExactSpelling = true)]
                extern static float *__MR_PointOnObject_Get_zBuffer(_Underlying *_this);
                return *__MR_PointOnObject_Get_zBuffer(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointOnObject() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointOnObject._Underlying *__MR_PointOnObject_DefaultConstruct();
            _UnderlyingPtr = __MR_PointOnObject_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointOnObject::PointOnObject`.
        public unsafe Const_PointOnObject(MR.Const_PointOnObject _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointOnObject._Underlying *__MR_PointOnObject_ConstructFromAnother(MR.PointOnObject._Underlying *_other);
            _UnderlyingPtr = __MR_PointOnObject_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PointOnObject::operator MR::PointOnFace`.
        public static unsafe implicit operator MR.PointOnFace(MR.Const_PointOnObject _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_ConvertTo_MR_PointOnFace", ExactSpelling = true)]
            extern static MR.PointOnFace._Underlying *__MR_PointOnObject_ConvertTo_MR_PointOnFace(MR.Const_PointOnObject._Underlying *_this);
            return new(__MR_PointOnObject_ConvertTo_MR_PointOnFace(_this._UnderlyingPtr), is_owning: true);
        }
    }

    // point located on either
    // 1. face of ObjectMesh
    // 2. line of ObjectLines
    // 3. point of ObjectPoints
    /// Generated from class `MR::PointOnObject`.
    /// This is the non-const half of the class.
    public class PointOnObject : Const_PointOnObject
    {
        internal unsafe PointOnObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// 3D location on the object in local coordinates
        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointOnObject_GetMutable_point(_Underlying *_this);
                return new(__MR_PointOnObject_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// z buffer value
        public new unsafe ref float ZBuffer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_GetMutable_zBuffer", ExactSpelling = true)]
                extern static float *__MR_PointOnObject_GetMutable_zBuffer(_Underlying *_this);
                return ref *__MR_PointOnObject_GetMutable_zBuffer(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointOnObject() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointOnObject._Underlying *__MR_PointOnObject_DefaultConstruct();
            _UnderlyingPtr = __MR_PointOnObject_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointOnObject::PointOnObject`.
        public unsafe PointOnObject(MR.Const_PointOnObject _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointOnObject._Underlying *__MR_PointOnObject_ConstructFromAnother(MR.PointOnObject._Underlying *_other);
            _UnderlyingPtr = __MR_PointOnObject_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointOnObject::operator=`.
        public unsafe MR.PointOnObject Assign(MR.Const_PointOnObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointOnObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointOnObject._Underlying *__MR_PointOnObject_AssignFromAnother(_Underlying *_this, MR.PointOnObject._Underlying *_other);
            return new(__MR_PointOnObject_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointOnObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointOnObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointOnObject`/`Const_PointOnObject` directly.
    public class _InOptMut_PointOnObject
    {
        public PointOnObject? Opt;

        public _InOptMut_PointOnObject() {}
        public _InOptMut_PointOnObject(PointOnObject value) {Opt = value;}
        public static implicit operator _InOptMut_PointOnObject(PointOnObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointOnObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointOnObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointOnObject`/`Const_PointOnObject` to pass it to the function.
    public class _InOptConst_PointOnObject
    {
        public Const_PointOnObject? Opt;

        public _InOptConst_PointOnObject() {}
        public _InOptConst_PointOnObject(Const_PointOnObject value) {Opt = value;}
        public static implicit operator _InOptConst_PointOnObject(Const_PointOnObject value) {return new(value);}
    }

    /// Converts PointOnObject coordinates depending on the object type to the PickedPoint variant
    /// Generated from function `MR::pointOnObjectToPickedPoint`.
    public static unsafe MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId PointOnObjectToPickedPoint(MR.Const_VisualObject? object_, MR.Const_PointOnObject pos)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointOnObjectToPickedPoint", ExactSpelling = true)]
        extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_pointOnObjectToPickedPoint(MR.Const_VisualObject._Underlying *object_, MR.Const_PointOnObject._Underlying *pos);
        return new(__MR_pointOnObjectToPickedPoint(object_ is not null ? object_._UnderlyingPtr : null, pos._UnderlyingPtr), is_owning: true);
    }

    /// Converts given point into local coordinates of its object,
    /// returns std::nullopt if object or point is invalid, or if it does not present in the object's topology
    /// Generated from function `MR::getPickedPointPosition`.
    public static unsafe MR.Std.Optional_MRVector3f GetPickedPointPosition(MR.Const_VisualObject object_, MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId point)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPickedPointPosition", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector3f._Underlying *__MR_getPickedPointPosition(MR.Const_VisualObject._Underlying *object_, MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *point);
        return new(__MR_getPickedPointPosition(object_._UnderlyingPtr, point._UnderlyingPtr), is_owning: true);
    }

    /// Returns object normal in local coordinates at given point,
    /// \param interpolated if true returns interpolated normal for mesh object, otherwise returns flat normal
    /// returns std::nullopt if object or point is invalid, or if it is ObjectLines or ObjectPoints without normals
    /// Generated from function `MR::getPickedPointNormal`.
    /// Parameter `interpolated` defaults to `true`.
    public static unsafe MR.Std.Optional_MRVector3f GetPickedPointNormal(MR.Const_VisualObject object_, MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId point, bool? interpolated = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPickedPointNormal", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector3f._Underlying *__MR_getPickedPointNormal(MR.Const_VisualObject._Underlying *object_, MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *point, byte *interpolated);
        byte __deref_interpolated = interpolated.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_getPickedPointNormal(object_._UnderlyingPtr, point._UnderlyingPtr, interpolated.HasValue ? &__deref_interpolated : null), is_owning: true);
    }
}

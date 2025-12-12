public static partial class MR
{
    /// This class can hold either ObjectMesh or ObjectPoint
    /// It is used for convenient storage and operation with any of them
    /// Generated from class `MR::MeshOrPointsObject`.
    /// This is the const half of the class.
    public class Const_MeshOrPointsObject : MR.Misc.Object, System.IDisposable, System.IEquatable<MR._ByValue_VisualObject>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOrPointsObject(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOrPointsObject_Destroy(_Underlying *_this);
            __MR_MeshOrPointsObject_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOrPointsObject() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshOrPointsObject() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshOrPointsObject_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe Const_MeshOrPointsObject(MR._ByValue_MeshOrPointsObject _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshOrPointsObject._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPointsObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe Const_MeshOrPointsObject(MR._ByValue_VisualObject vo) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject(MR.Misc._PassBy vo_pass_by, MR.VisualObject._UnderlyingShared *vo);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject(vo.PassByMode, vo.Value is not null ? vo.Value._UnderlyingSharedPtr : null);
        }

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator Const_MeshOrPointsObject(MR._ByValue_VisualObject vo) {return new(vo);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe Const_MeshOrPointsObject(MR._ByValue_ObjectMesh om) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh(MR.Misc._PassBy om_pass_by, MR.ObjectMesh._UnderlyingShared *om);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh(om.PassByMode, om.Value is not null ? om.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator Const_MeshOrPointsObject(MR._ByValue_ObjectMesh om) {return new(om);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe Const_MeshOrPointsObject(MR._ByValue_ObjectPoints op) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints(MR.Misc._PassBy op_pass_by, MR.ObjectPoints._UnderlyingShared *op);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints(op.PassByMode, op.Value is not null ? op.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator Const_MeshOrPointsObject(MR._ByValue_ObjectPoints op) {return new(op);}

        /// if holding ObjectMesh, return pointer to it, otherwise return nullptr
        /// Generated from method `MR::MeshOrPointsObject::asObjectMesh`.
        public unsafe MR.ObjectMesh? AsObjectMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_asObjectMesh", ExactSpelling = true)]
            extern static MR.ObjectMesh._Underlying *__MR_MeshOrPointsObject_asObjectMesh(_Underlying *_this);
            var __ret = __MR_MeshOrPointsObject_asObjectMesh(_UnderlyingPtr);
            return __ret is not null ? new MR.ObjectMesh(__ret, is_owning: false) : null;
        }

        /// if holding ObjectPoints, return pointer to it, otherwise return nullptr
        /// Generated from method `MR::MeshOrPointsObject::asObjectPoints`.
        public unsafe MR.ObjectPoints? AsObjectPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_asObjectPoints", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_MeshOrPointsObject_asObjectPoints(_Underlying *_this);
            var __ret = __MR_MeshOrPointsObject_asObjectPoints(_UnderlyingPtr);
            return __ret is not null ? new MR.ObjectPoints(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::MeshOrPointsObject::operator->`.
        public unsafe MR.Const_VisualObject Arrow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_arrow", ExactSpelling = true)]
            extern static MR.Const_VisualObject._UnderlyingShared *__MR_MeshOrPointsObject_arrow(_Underlying *_this);
            return new(__MR_MeshOrPointsObject_arrow(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOrPointsObject::get`.
        public unsafe MR.Const_VisualObject Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_get", ExactSpelling = true)]
            extern static MR.Const_VisualObject._UnderlyingShared *__MR_MeshOrPointsObject_get(_Underlying *_this);
            return new(__MR_MeshOrPointsObject_get(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOrPointsObject::operator==`.
        public static unsafe bool operator==(MR.Const_MeshOrPointsObject _this, MR._ByValue_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_MeshOrPointsObject_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
            extern static byte __MR_equal_MR_MeshOrPointsObject_std_shared_ptr_MR_VisualObject(MR.Const_MeshOrPointsObject._Underlying *_this, MR.Misc._PassBy other_pass_by, MR.VisualObject._UnderlyingShared *other);
            return __MR_equal_MR_MeshOrPointsObject_std_shared_ptr_MR_VisualObject(_this._UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingSharedPtr : null) != 0;
        }

        public static unsafe bool operator!=(MR.Const_MeshOrPointsObject _this, MR._ByValue_VisualObject other)
        {
            return !(_this == other);
        }

        /// get class that hold either mesh part or point cloud
        /// Generated from method `MR::MeshOrPointsObject::meshOrPoints`.
        public unsafe MR.MeshOrPoints MeshOrPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_meshOrPoints", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPointsObject_meshOrPoints(_Underlying *_this);
            return new(__MR_MeshOrPointsObject_meshOrPoints(_UnderlyingPtr), is_owning: true);
        }

        // IEquatable:

        public bool Equals(MR._ByValue_VisualObject? other)
        {
            if (other is null)
                return false;
            return this == other;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR._ByValue_VisualObject)
                return this == (MR._ByValue_VisualObject)other;
            return false;
        }
    }

    /// This class can hold either ObjectMesh or ObjectPoint
    /// It is used for convenient storage and operation with any of them
    /// Generated from class `MR::MeshOrPointsObject`.
    /// This is the non-const half of the class.
    public class MeshOrPointsObject : Const_MeshOrPointsObject
    {
        internal unsafe MeshOrPointsObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshOrPointsObject() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshOrPointsObject_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe MeshOrPointsObject(MR._ByValue_MeshOrPointsObject _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshOrPointsObject._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPointsObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe MeshOrPointsObject(MR._ByValue_VisualObject vo) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject(MR.Misc._PassBy vo_pass_by, MR.VisualObject._UnderlyingShared *vo);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_VisualObject(vo.PassByMode, vo.Value is not null ? vo.Value._UnderlyingSharedPtr : null);
        }

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator MeshOrPointsObject(MR._ByValue_VisualObject vo) {return new(vo);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe MeshOrPointsObject(MR._ByValue_ObjectMesh om) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh(MR.Misc._PassBy om_pass_by, MR.ObjectMesh._UnderlyingShared *om);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectMesh(om.PassByMode, om.Value is not null ? om.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator MeshOrPointsObject(MR._ByValue_ObjectMesh om) {return new(om);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public unsafe MeshOrPointsObject(MR._ByValue_ObjectPoints op) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints(MR.Misc._PassBy op_pass_by, MR.ObjectPoints._UnderlyingShared *op);
            _UnderlyingPtr = __MR_MeshOrPointsObject_Construct_std_shared_ptr_MR_ObjectPoints(op.PassByMode, op.Value is not null ? op.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator MeshOrPointsObject(MR._ByValue_ObjectPoints op) {return new(op);}

        /// Generated from method `MR::MeshOrPointsObject::operator=`.
        public unsafe MR.MeshOrPointsObject Assign(MR._ByValue_MeshOrPointsObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsObject._Underlying *__MR_MeshOrPointsObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshOrPointsObject._Underlying *_other);
            return new(__MR_MeshOrPointsObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set to hold ObjectMesh
        /// Generated from method `MR::MeshOrPointsObject::set`.
        public unsafe void Set(MR._ByValue_ObjectMesh om)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
            extern static void __MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectMesh(_Underlying *_this, MR.Misc._PassBy om_pass_by, MR.ObjectMesh._UnderlyingShared *om);
            __MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectMesh(_UnderlyingPtr, om.PassByMode, om.Value is not null ? om.Value._UnderlyingSharedPtr : null);
        }

        /// set to hold ObjectPoints
        /// Generated from method `MR::MeshOrPointsObject::set`.
        public unsafe void Set(MR._ByValue_ObjectPoints op)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
            extern static void __MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectPoints(_Underlying *_this, MR.Misc._PassBy op_pass_by, MR.ObjectPoints._UnderlyingShared *op);
            __MR_MeshOrPointsObject_set_std_shared_ptr_MR_ObjectPoints(_UnderlyingPtr, op.PassByMode, op.Value is not null ? op.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from method `MR::MeshOrPointsObject::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsObject_reset", ExactSpelling = true)]
            extern static void __MR_MeshOrPointsObject_reset(_Underlying *_this);
            __MR_MeshOrPointsObject_reset(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOrPointsObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOrPointsObject`/`Const_MeshOrPointsObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOrPointsObject
    {
        internal readonly Const_MeshOrPointsObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOrPointsObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshOrPointsObject(Const_MeshOrPointsObject new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOrPointsObject(Const_MeshOrPointsObject arg) {return new(arg);}
        public _ByValue_MeshOrPointsObject(MR.Misc._Moved<MeshOrPointsObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshOrPointsObject(MR.Misc._Moved<MeshOrPointsObject> arg) {return new(arg);}

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _ByValue_MeshOrPointsObject(MR._ByValue_VisualObject vo) {return new MR.MeshOrPointsObject(vo);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _ByValue_MeshOrPointsObject(MR._ByValue_ObjectMesh om) {return new MR.MeshOrPointsObject(om);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _ByValue_MeshOrPointsObject(MR._ByValue_ObjectPoints op) {return new MR.MeshOrPointsObject(op);}
    }

    /// This is used for optional parameters of class `MeshOrPointsObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOrPointsObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPointsObject`/`Const_MeshOrPointsObject` directly.
    public class _InOptMut_MeshOrPointsObject
    {
        public MeshOrPointsObject? Opt;

        public _InOptMut_MeshOrPointsObject() {}
        public _InOptMut_MeshOrPointsObject(MeshOrPointsObject value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOrPointsObject(MeshOrPointsObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOrPointsObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOrPointsObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPointsObject`/`Const_MeshOrPointsObject` to pass it to the function.
    public class _InOptConst_MeshOrPointsObject
    {
        public Const_MeshOrPointsObject? Opt;

        public _InOptConst_MeshOrPointsObject() {}
        public _InOptConst_MeshOrPointsObject(Const_MeshOrPointsObject value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOrPointsObject(Const_MeshOrPointsObject value) {return new(value);}

        /// construct, automatically detecting the object type (ObjectMesh or ObjectPoint)
        /// if set an another type, will be reset
        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _InOptConst_MeshOrPointsObject(MR._ByValue_VisualObject vo) {return new MR.MeshOrPointsObject(vo);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _InOptConst_MeshOrPointsObject(MR._ByValue_ObjectMesh om) {return new MR.MeshOrPointsObject(om);}

        /// Generated from constructor `MR::MeshOrPointsObject::MeshOrPointsObject`.
        public static unsafe implicit operator _InOptConst_MeshOrPointsObject(MR._ByValue_ObjectPoints op) {return new MR.MeshOrPointsObject(op);}
    }
}

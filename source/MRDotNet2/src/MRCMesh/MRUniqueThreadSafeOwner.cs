public static partial class MR
{
    /// This class is base class for unique thread safe owning of some objects, for example AABBTree
    /// classes derived from this one should have function like getOrCreate
    /// Generated from class `MR::UniqueThreadSafeOwner<MR::AABBTree>`.
    /// This is the const half of the class.
    public class Const_UniqueThreadSafeOwner_MRAABBTree : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UniqueThreadSafeOwner_MRAABBTree(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_Destroy", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTree_Destroy(_Underlying *_this);
            __MR_UniqueThreadSafeOwner_MR_AABBTree_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UniqueThreadSafeOwner_MRAABBTree() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UniqueThreadSafeOwner_MRAABBTree() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct();
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct();
        }

        /// Generated from constructor `MR::UniqueThreadSafeOwner<MR::AABBTree>::UniqueThreadSafeOwner`.
        public unsafe Const_UniqueThreadSafeOwner_MRAABBTree(MR._ByValue_UniqueThreadSafeOwner_MRAABBTree _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_UniqueThreadSafeOwner_MR_AABBTree_heapBytes(_Underlying *_this);
            return __MR_UniqueThreadSafeOwner_MR_AABBTree_heapBytes(_UnderlyingPtr);
        }
    }

    /// This class is base class for unique thread safe owning of some objects, for example AABBTree
    /// classes derived from this one should have function like getOrCreate
    /// Generated from class `MR::UniqueThreadSafeOwner<MR::AABBTree>`.
    /// This is the non-const half of the class.
    public class UniqueThreadSafeOwner_MRAABBTree : Const_UniqueThreadSafeOwner_MRAABBTree
    {
        internal unsafe UniqueThreadSafeOwner_MRAABBTree(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UniqueThreadSafeOwner_MRAABBTree() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct();
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTree_DefaultConstruct();
        }

        /// Generated from constructor `MR::UniqueThreadSafeOwner<MR::AABBTree>::UniqueThreadSafeOwner`.
        public unsafe UniqueThreadSafeOwner_MRAABBTree(MR._ByValue_UniqueThreadSafeOwner_MRAABBTree _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTree_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::operator=`.
        public unsafe MR.UniqueThreadSafeOwner_MRAABBTree Assign(MR._ByValue_UniqueThreadSafeOwner_MRAABBTree b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy b_pass_by, MR.UniqueThreadSafeOwner_MRAABBTree._Underlying *b);
            return new(__MR_UniqueThreadSafeOwner_MR_AABBTree_AssignFromAnother(_UnderlyingPtr, b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// deletes owned object
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_reset", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTree_reset(_Underlying *_this);
            __MR_UniqueThreadSafeOwner_MR_AABBTree_reset(_UnderlyingPtr);
        }

        /// returns existing owned object and does not create new one
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::get`.
        public unsafe MR.AABBTree? Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_get", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_get(_Underlying *_this);
            var __ret = __MR_UniqueThreadSafeOwner_MR_AABBTree_get(_UnderlyingPtr);
            return __ret is not null ? new MR.AABBTree(__ret, is_owning: false) : null;
        }

        /// returns existing owned object or creates new one using creator function
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::getOrCreate`.
        public unsafe MR.AABBTree GetOrCreate(MR.Std.Const_Function_MRAABBTreeFunc creator)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_getOrCreate", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTree_getOrCreate(_Underlying *_this, MR.Std.Const_Function_MRAABBTreeFunc._Underlying *creator);
            return new(__MR_UniqueThreadSafeOwner_MR_AABBTree_getOrCreate(_UnderlyingPtr, creator._UnderlyingPtr), is_owning: false);
        }

        /// calls given updater for the owned object (if any)
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTree>::update`.
        public unsafe void Update(MR.Std.Const_Function_VoidFuncFromMRAABBTreeRef updater)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTree_update", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTree_update(_Underlying *_this, MR.Std.Const_Function_VoidFuncFromMRAABBTreeRef._Underlying *updater);
            __MR_UniqueThreadSafeOwner_MR_AABBTree_update(_UnderlyingPtr, updater._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UniqueThreadSafeOwner_MRAABBTree` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTree`/`Const_UniqueThreadSafeOwner_MRAABBTree` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UniqueThreadSafeOwner_MRAABBTree
    {
        internal readonly Const_UniqueThreadSafeOwner_MRAABBTree? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UniqueThreadSafeOwner_MRAABBTree() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UniqueThreadSafeOwner_MRAABBTree(Const_UniqueThreadSafeOwner_MRAABBTree new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UniqueThreadSafeOwner_MRAABBTree(Const_UniqueThreadSafeOwner_MRAABBTree arg) {return new(arg);}
        public _ByValue_UniqueThreadSafeOwner_MRAABBTree(MR.Misc._Moved<UniqueThreadSafeOwner_MRAABBTree> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UniqueThreadSafeOwner_MRAABBTree(MR.Misc._Moved<UniqueThreadSafeOwner_MRAABBTree> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UniqueThreadSafeOwner_MRAABBTree` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UniqueThreadSafeOwner_MRAABBTree`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTree`/`Const_UniqueThreadSafeOwner_MRAABBTree` directly.
    public class _InOptMut_UniqueThreadSafeOwner_MRAABBTree
    {
        public UniqueThreadSafeOwner_MRAABBTree? Opt;

        public _InOptMut_UniqueThreadSafeOwner_MRAABBTree() {}
        public _InOptMut_UniqueThreadSafeOwner_MRAABBTree(UniqueThreadSafeOwner_MRAABBTree value) {Opt = value;}
        public static implicit operator _InOptMut_UniqueThreadSafeOwner_MRAABBTree(UniqueThreadSafeOwner_MRAABBTree value) {return new(value);}
    }

    /// This is used for optional parameters of class `UniqueThreadSafeOwner_MRAABBTree` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UniqueThreadSafeOwner_MRAABBTree`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTree`/`Const_UniqueThreadSafeOwner_MRAABBTree` to pass it to the function.
    public class _InOptConst_UniqueThreadSafeOwner_MRAABBTree
    {
        public Const_UniqueThreadSafeOwner_MRAABBTree? Opt;

        public _InOptConst_UniqueThreadSafeOwner_MRAABBTree() {}
        public _InOptConst_UniqueThreadSafeOwner_MRAABBTree(Const_UniqueThreadSafeOwner_MRAABBTree value) {Opt = value;}
        public static implicit operator _InOptConst_UniqueThreadSafeOwner_MRAABBTree(Const_UniqueThreadSafeOwner_MRAABBTree value) {return new(value);}
    }

    /// This class is base class for unique thread safe owning of some objects, for example AABBTree
    /// classes derived from this one should have function like getOrCreate
    /// Generated from class `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>`.
    /// This is the const half of the class.
    public class Const_UniqueThreadSafeOwner_MRAABBTreePoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UniqueThreadSafeOwner_MRAABBTreePoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_Destroy", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_Destroy(_Underlying *_this);
            __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UniqueThreadSafeOwner_MRAABBTreePoints() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UniqueThreadSafeOwner_MRAABBTreePoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct();
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct();
        }

        /// Generated from constructor `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::UniqueThreadSafeOwner`.
        public unsafe Const_UniqueThreadSafeOwner_MRAABBTreePoints(MR._ByValue_UniqueThreadSafeOwner_MRAABBTreePoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_heapBytes(_Underlying *_this);
            return __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_heapBytes(_UnderlyingPtr);
        }
    }

    /// This class is base class for unique thread safe owning of some objects, for example AABBTree
    /// classes derived from this one should have function like getOrCreate
    /// Generated from class `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>`.
    /// This is the non-const half of the class.
    public class UniqueThreadSafeOwner_MRAABBTreePoints : Const_UniqueThreadSafeOwner_MRAABBTreePoints
    {
        internal unsafe UniqueThreadSafeOwner_MRAABBTreePoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UniqueThreadSafeOwner_MRAABBTreePoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct();
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_DefaultConstruct();
        }

        /// Generated from constructor `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::UniqueThreadSafeOwner`.
        public unsafe UniqueThreadSafeOwner_MRAABBTreePoints(MR._ByValue_UniqueThreadSafeOwner_MRAABBTreePoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::operator=`.
        public unsafe MR.UniqueThreadSafeOwner_MRAABBTreePoints Assign(MR._ByValue_UniqueThreadSafeOwner_MRAABBTreePoints b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy b_pass_by, MR.UniqueThreadSafeOwner_MRAABBTreePoints._Underlying *b);
            return new(__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_AssignFromAnother(_UnderlyingPtr, b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// deletes owned object
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_reset", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_reset(_Underlying *_this);
            __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_reset(_UnderlyingPtr);
        }

        /// returns existing owned object and does not create new one
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::get`.
        public unsafe MR.AABBTreePoints? Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_get", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_get(_Underlying *_this);
            var __ret = __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_get(_UnderlyingPtr);
            return __ret is not null ? new MR.AABBTreePoints(__ret, is_owning: false) : null;
        }

        /// returns existing owned object or creates new one using creator function
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::getOrCreate`.
        public unsafe MR.AABBTreePoints GetOrCreate(MR.Std.Const_Function_MRAABBTreePointsFunc creator)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_getOrCreate", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_getOrCreate(_Underlying *_this, MR.Std.Const_Function_MRAABBTreePointsFunc._Underlying *creator);
            return new(__MR_UniqueThreadSafeOwner_MR_AABBTreePoints_getOrCreate(_UnderlyingPtr, creator._UnderlyingPtr), is_owning: false);
        }

        /// calls given updater for the owned object (if any)
        /// Generated from method `MR::UniqueThreadSafeOwner<MR::AABBTreePoints>::update`.
        public unsafe void Update(MR.Std.Const_Function_VoidFuncFromMRAABBTreePointsRef updater)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueThreadSafeOwner_MR_AABBTreePoints_update", ExactSpelling = true)]
            extern static void __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_update(_Underlying *_this, MR.Std.Const_Function_VoidFuncFromMRAABBTreePointsRef._Underlying *updater);
            __MR_UniqueThreadSafeOwner_MR_AABBTreePoints_update(_UnderlyingPtr, updater._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UniqueThreadSafeOwner_MRAABBTreePoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTreePoints`/`Const_UniqueThreadSafeOwner_MRAABBTreePoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints
    {
        internal readonly Const_UniqueThreadSafeOwner_MRAABBTreePoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints(Const_UniqueThreadSafeOwner_MRAABBTreePoints new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints(Const_UniqueThreadSafeOwner_MRAABBTreePoints arg) {return new(arg);}
        public _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints(MR.Misc._Moved<UniqueThreadSafeOwner_MRAABBTreePoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UniqueThreadSafeOwner_MRAABBTreePoints(MR.Misc._Moved<UniqueThreadSafeOwner_MRAABBTreePoints> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UniqueThreadSafeOwner_MRAABBTreePoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UniqueThreadSafeOwner_MRAABBTreePoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTreePoints`/`Const_UniqueThreadSafeOwner_MRAABBTreePoints` directly.
    public class _InOptMut_UniqueThreadSafeOwner_MRAABBTreePoints
    {
        public UniqueThreadSafeOwner_MRAABBTreePoints? Opt;

        public _InOptMut_UniqueThreadSafeOwner_MRAABBTreePoints() {}
        public _InOptMut_UniqueThreadSafeOwner_MRAABBTreePoints(UniqueThreadSafeOwner_MRAABBTreePoints value) {Opt = value;}
        public static implicit operator _InOptMut_UniqueThreadSafeOwner_MRAABBTreePoints(UniqueThreadSafeOwner_MRAABBTreePoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `UniqueThreadSafeOwner_MRAABBTreePoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UniqueThreadSafeOwner_MRAABBTreePoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueThreadSafeOwner_MRAABBTreePoints`/`Const_UniqueThreadSafeOwner_MRAABBTreePoints` to pass it to the function.
    public class _InOptConst_UniqueThreadSafeOwner_MRAABBTreePoints
    {
        public Const_UniqueThreadSafeOwner_MRAABBTreePoints? Opt;

        public _InOptConst_UniqueThreadSafeOwner_MRAABBTreePoints() {}
        public _InOptConst_UniqueThreadSafeOwner_MRAABBTreePoints(Const_UniqueThreadSafeOwner_MRAABBTreePoints value) {Opt = value;}
        public static implicit operator _InOptConst_UniqueThreadSafeOwner_MRAABBTreePoints(Const_UniqueThreadSafeOwner_MRAABBTreePoints value) {return new(value);}
    }
}

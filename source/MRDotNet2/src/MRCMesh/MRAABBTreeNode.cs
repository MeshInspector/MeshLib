public static partial class MR
{
    /// Generated from class `MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>`.
    /// This is the const half of the class.
    public class Const_AABBTreeTraits_MRFaceTag_MRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeTraits_MRFaceTag_MRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeTraits_MRFaceTag_MRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeTraits_MRFaceTag_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>::AABBTreeTraits`.
        public unsafe Const_AABBTreeTraits_MRFaceTag_MRBox3f(MR.Const_AABBTreeTraits_MRFaceTag_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>`.
    /// This is the non-const half of the class.
    public class AABBTreeTraits_MRFaceTag_MRBox3f : Const_AABBTreeTraits_MRFaceTag_MRBox3f
    {
        internal unsafe AABBTreeTraits_MRFaceTag_MRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeTraits_MRFaceTag_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>::AABBTreeTraits`.
        public unsafe AABBTreeTraits_MRFaceTag_MRBox3f(MR.Const_AABBTreeTraits_MRFaceTag_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>::operator=`.
        public unsafe MR.AABBTreeTraits_MRFaceTag_MRBox3f Assign(MR.Const_AABBTreeTraits_MRFaceTag_MRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.AABBTreeTraits_MRFaceTag_MRBox3f._Underlying *_other);
            return new(__MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRFaceTag_MRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeTraits_MRFaceTag_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRFaceTag_MRBox3f`/`Const_AABBTreeTraits_MRFaceTag_MRBox3f` directly.
    public class _InOptMut_AABBTreeTraits_MRFaceTag_MRBox3f
    {
        public AABBTreeTraits_MRFaceTag_MRBox3f? Opt;

        public _InOptMut_AABBTreeTraits_MRFaceTag_MRBox3f() {}
        public _InOptMut_AABBTreeTraits_MRFaceTag_MRBox3f(AABBTreeTraits_MRFaceTag_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeTraits_MRFaceTag_MRBox3f(AABBTreeTraits_MRFaceTag_MRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRFaceTag_MRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeTraits_MRFaceTag_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRFaceTag_MRBox3f`/`Const_AABBTreeTraits_MRFaceTag_MRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeTraits_MRFaceTag_MRBox3f
    {
        public Const_AABBTreeTraits_MRFaceTag_MRBox3f? Opt;

        public _InOptConst_AABBTreeTraits_MRFaceTag_MRBox3f() {}
        public _InOptConst_AABBTreeTraits_MRFaceTag_MRBox3f(Const_AABBTreeTraits_MRFaceTag_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeTraits_MRFaceTag_MRBox3f(Const_AABBTreeTraits_MRFaceTag_MRBox3f value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>`.
    /// This is the const half of the class.
    public class Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_Underlying *_this);
            __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>::AABBTreeTraits`.
        public unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>`.
    /// This is the non-const half of the class.
    public class AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f : Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f
    {
        internal unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>::AABBTreeTraits`.
        public unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>::operator=`.
        public unsafe MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f Assign(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_Underlying *_this, MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f._Underlying *_other);
            return new(__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f`/`Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f` directly.
    public class _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f
    {
        public AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f? Opt;

        public _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f() {}
        public _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f`/`Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f` to pass it to the function.
    public class _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f
    {
        public Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f? Opt;

        public _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f() {}
        public _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f(Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox2f value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>`.
    /// This is the const half of the class.
    public class Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>::AABBTreeTraits`.
        public unsafe Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>`.
    /// This is the non-const half of the class.
    public class AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f : Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f
    {
        internal unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>::AABBTreeTraits`.
        public unsafe AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>::operator=`.
        public unsafe MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f Assign(MR.Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f._Underlying *_other);
            return new(__MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f`/`Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f` directly.
    public class _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f
    {
        public AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f? Opt;

        public _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f() {}
        public _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f`/`Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f
    {
        public Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f? Opt;

        public _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f() {}
        public _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f(Const_AABBTreeTraits_MRUndirectedEdgeTag_MRBox3f value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`.
    /// This is the const half of the class.
    public class Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f() {Dispose(false);}

        ///< bounding box of whole subtree
        public unsafe MR.Const_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_box", ExactSpelling = true)]
                extern static MR.Const_Box3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_l", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_r", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Get_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>` elementwise.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::AABBTreeNode`.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if this is a leaf node without children nodes but with a LeafId reference
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::leaf`.
        public unsafe bool Leaf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leaf", ExactSpelling = true)]
            extern static byte __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leaf(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leaf(_UnderlyingPtr) != 0;
        }

        /// returns face (for the leaf node only)
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::leafId`.
        public unsafe MR.FaceId LeafId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leafId", ExactSpelling = true)]
            extern static MR.FaceId __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leafId(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_leafId(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`.
    /// This is the non-const half of the class.
    public class AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f : Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        internal unsafe AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< bounding box of whole subtree
        public new unsafe MR.Mut_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_box", ExactSpelling = true)]
                extern static MR.Mut_Box3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_l", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_r", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_GetMutable_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>` elementwise.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::AABBTreeNode`.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::operator=`.
        public unsafe MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f Assign(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::setLeafId`.
        public unsafe void SetLeafId(MR.FaceId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_setLeafId", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_setLeafId(_Underlying *_this, MR.FaceId id);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_setLeafId(_UnderlyingPtr, id);
        }
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f` directly.
    public class _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        public AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f? Opt;

        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f() {}
        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        public Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f? Opt;

        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f() {}
        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeNode<MR::ObjTreeTraits>`.
    /// This is the const half of the class.
    public class Const_AABBTreeNode_MRObjTreeTraits : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeNode_MRObjTreeTraits(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_ObjTreeTraits_Destroy(_Underlying *_this);
            __MR_AABBTreeNode_MR_ObjTreeTraits_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeNode_MRObjTreeTraits() {Dispose(false);}

        ///< bounding box of whole subtree
        public unsafe MR.Const_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_Get_box", ExactSpelling = true)]
                extern static MR.Const_Box3f._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_Get_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_Get_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_Get_l", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_Get_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_Get_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_Get_r", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_Get_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_Get_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeNode_MRObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::ObjTreeTraits>` elementwise.
        public unsafe Const_AABBTreeNode_MRObjTreeTraits(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::ObjTreeTraits>::AABBTreeNode`.
        public unsafe Const_AABBTreeNode_MRObjTreeTraits(MR.Const_AABBTreeNode_MRObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother(MR.AABBTreeNode_MRObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if this is a leaf node without children nodes but with a LeafId reference
        /// Generated from method `MR::AABBTreeNode<MR::ObjTreeTraits>::leaf`.
        public unsafe bool Leaf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_leaf", ExactSpelling = true)]
            extern static byte __MR_AABBTreeNode_MR_ObjTreeTraits_leaf(_Underlying *_this);
            return __MR_AABBTreeNode_MR_ObjTreeTraits_leaf(_UnderlyingPtr) != 0;
        }

        /// returns face (for the leaf node only)
        /// Generated from method `MR::AABBTreeNode<MR::ObjTreeTraits>::leafId`.
        public unsafe MR.ObjId LeafId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_leafId", ExactSpelling = true)]
            extern static MR.ObjId __MR_AABBTreeNode_MR_ObjTreeTraits_leafId(_Underlying *_this);
            return __MR_AABBTreeNode_MR_ObjTreeTraits_leafId(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeNode<MR::ObjTreeTraits>`.
    /// This is the non-const half of the class.
    public class AABBTreeNode_MRObjTreeTraits : Const_AABBTreeNode_MRObjTreeTraits
    {
        internal unsafe AABBTreeNode_MRObjTreeTraits(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< bounding box of whole subtree
        public new unsafe MR.Mut_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_box", ExactSpelling = true)]
                extern static MR.Mut_Box3f._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_l", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_r", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_ObjTreeTraits_GetMutable_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeNode_MRObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::ObjTreeTraits>` elementwise.
        public unsafe AABBTreeNode_MRObjTreeTraits(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::ObjTreeTraits>::AABBTreeNode`.
        public unsafe AABBTreeNode_MRObjTreeTraits(MR.Const_AABBTreeNode_MRObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother(MR.AABBTreeNode_MRObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_ObjTreeTraits_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeNode<MR::ObjTreeTraits>::operator=`.
        public unsafe MR.AABBTreeNode_MRObjTreeTraits Assign(MR.Const_AABBTreeNode_MRObjTreeTraits _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeNode_MR_ObjTreeTraits_AssignFromAnother(_Underlying *_this, MR.AABBTreeNode_MRObjTreeTraits._Underlying *_other);
            return new(__MR_AABBTreeNode_MR_ObjTreeTraits_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AABBTreeNode<MR::ObjTreeTraits>::setLeafId`.
        public unsafe void SetLeafId(MR.ObjId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_ObjTreeTraits_setLeafId", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_ObjTreeTraits_setLeafId(_Underlying *_this, MR.ObjId id);
            __MR_AABBTreeNode_MR_ObjTreeTraits_setLeafId(_UnderlyingPtr, id);
        }
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRObjTreeTraits` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeNode_MRObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRObjTreeTraits`/`Const_AABBTreeNode_MRObjTreeTraits` directly.
    public class _InOptMut_AABBTreeNode_MRObjTreeTraits
    {
        public AABBTreeNode_MRObjTreeTraits? Opt;

        public _InOptMut_AABBTreeNode_MRObjTreeTraits() {}
        public _InOptMut_AABBTreeNode_MRObjTreeTraits(AABBTreeNode_MRObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeNode_MRObjTreeTraits(AABBTreeNode_MRObjTreeTraits value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRObjTreeTraits` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeNode_MRObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRObjTreeTraits`/`Const_AABBTreeNode_MRObjTreeTraits` to pass it to the function.
    public class _InOptConst_AABBTreeNode_MRObjTreeTraits
    {
        public Const_AABBTreeNode_MRObjTreeTraits? Opt;

        public _InOptConst_AABBTreeNode_MRObjTreeTraits() {}
        public _InOptConst_AABBTreeNode_MRObjTreeTraits(Const_AABBTreeNode_MRObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeNode_MRObjTreeTraits(Const_AABBTreeNode_MRObjTreeTraits value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`.
    /// This is the const half of the class.
    public class Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_Underlying *_this);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {Dispose(false);}

        ///< bounding box of whole subtree
        public unsafe MR.Const_Box2f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_box", ExactSpelling = true)]
                extern static MR.Const_Box2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_l", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_r", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Get_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>` elementwise.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Box2f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom(MR.Box2f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::AABBTreeNode`.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if this is a leaf node without children nodes but with a LeafId reference
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::leaf`.
        public unsafe bool Leaf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leaf", ExactSpelling = true)]
            extern static byte __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leaf(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leaf(_UnderlyingPtr) != 0;
        }

        /// returns face (for the leaf node only)
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::leafId`.
        public unsafe MR.UndirectedEdgeId LeafId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leafId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leafId(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_leafId(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`.
    /// This is the non-const half of the class.
    public class AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f : Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        internal unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< bounding box of whole subtree
        public new unsafe MR.Mut_Box2f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_box", ExactSpelling = true)]
                extern static MR.Mut_Box2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_l", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_r", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_GetMutable_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>` elementwise.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Box2f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom(MR.Box2f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::AABBTreeNode`.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::operator=`.
        public unsafe MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f Assign(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_Underlying *_this, MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::setLeafId`.
        public unsafe void SetLeafId(MR.UndirectedEdgeId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_setLeafId", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_setLeafId(_Underlying *_this, MR.UndirectedEdgeId id);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_setLeafId(_UnderlyingPtr, id);
        }
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` directly.
    public class _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        public AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f? Opt;

        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {}
        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` to pass it to the function.
    public class _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        public Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f? Opt;

        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {}
        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {return new(value);}
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`.
    /// This is the const half of the class.
    public class Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {Dispose(false);}

        ///< bounding box of whole subtree
        public unsafe MR.Const_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_box", ExactSpelling = true)]
                extern static MR.Const_Box3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_l", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public unsafe MR.Const_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_r", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Get_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>` elementwise.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::AABBTreeNode`.
        public unsafe Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if this is a leaf node without children nodes but with a LeafId reference
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::leaf`.
        public unsafe bool Leaf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leaf", ExactSpelling = true)]
            extern static byte __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leaf(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leaf(_UnderlyingPtr) != 0;
        }

        /// returns face (for the leaf node only)
        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::leafId`.
        public unsafe MR.UndirectedEdgeId LeafId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leafId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leafId(_Underlying *_this);
            return __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_leafId(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`.
    /// This is the non-const half of the class.
    public class AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f : Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        internal unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< bounding box of whole subtree
        public new unsafe MR.Mut_Box3f Box
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_box", ExactSpelling = true)]
                extern static MR.Mut_Box3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_box(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_box(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId L
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_l", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_l(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_l(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< two children
        public new unsafe MR.Mut_NodeId R
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_r", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_r(_Underlying *_this);
                return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_GetMutable_r(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Constructs `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>` elementwise.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFrom(box, l, r);
        }

        /// Generated from constructor `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::AABBTreeNode`.
        public unsafe AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::operator=`.
        public unsafe MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f Assign(MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            return new(__MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AABBTreeNode<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::setLeafId`.
        public unsafe void SetLeafId(MR.UndirectedEdgeId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_setLeafId", ExactSpelling = true)]
            extern static void __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_setLeafId(_Underlying *_this, MR.UndirectedEdgeId id);
            __MR_AABBTreeNode_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_setLeafId(_UnderlyingPtr, id);
        }
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` directly.
    public class _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        public AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f? Opt;

        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {}
        public _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`/`Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        public Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f? Opt;

        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {}
        public _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {return new(value);}
    }

    /// Generated from class `MR::NodeNode`.
    /// This is the const half of the class.
    public class Const_NodeNode : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NodeNode(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_Destroy", ExactSpelling = true)]
            extern static void __MR_NodeNode_Destroy(_Underlying *_this);
            __MR_NodeNode_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NodeNode() {Dispose(false);}

        public unsafe MR.Const_NodeId ANode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_Get_aNode", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_NodeNode_Get_aNode(_Underlying *_this);
                return new(__MR_NodeNode_Get_aNode(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_NodeId BNode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_Get_bNode", ExactSpelling = true)]
                extern static MR.Const_NodeId._Underlying *__MR_NodeNode_Get_bNode(_Underlying *_this);
                return new(__MR_NodeNode_Get_bNode(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NodeNode() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_DefaultConstruct();
            _UnderlyingPtr = __MR_NodeNode_DefaultConstruct();
        }

        /// Constructs `MR::NodeNode` elementwise.
        public unsafe Const_NodeNode(MR.NodeId aNode, MR.NodeId bNode) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_ConstructFrom", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_ConstructFrom(MR.NodeId aNode, MR.NodeId bNode);
            _UnderlyingPtr = __MR_NodeNode_ConstructFrom(aNode, bNode);
        }

        /// Generated from constructor `MR::NodeNode::NodeNode`.
        public unsafe Const_NodeNode(MR.Const_NodeNode _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_ConstructFromAnother(MR.NodeNode._Underlying *_other);
            _UnderlyingPtr = __MR_NodeNode_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::NodeNode`.
    /// This is the non-const half of the class.
    public class NodeNode : Const_NodeNode
    {
        internal unsafe NodeNode(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_NodeId ANode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_GetMutable_aNode", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_NodeNode_GetMutable_aNode(_Underlying *_this);
                return new(__MR_NodeNode_GetMutable_aNode(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_NodeId BNode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_GetMutable_bNode", ExactSpelling = true)]
                extern static MR.Mut_NodeId._Underlying *__MR_NodeNode_GetMutable_bNode(_Underlying *_this);
                return new(__MR_NodeNode_GetMutable_bNode(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NodeNode() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_DefaultConstruct();
            _UnderlyingPtr = __MR_NodeNode_DefaultConstruct();
        }

        /// Constructs `MR::NodeNode` elementwise.
        public unsafe NodeNode(MR.NodeId aNode, MR.NodeId bNode) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_ConstructFrom", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_ConstructFrom(MR.NodeId aNode, MR.NodeId bNode);
            _UnderlyingPtr = __MR_NodeNode_ConstructFrom(aNode, bNode);
        }

        /// Generated from constructor `MR::NodeNode::NodeNode`.
        public unsafe NodeNode(MR.Const_NodeNode _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_ConstructFromAnother(MR.NodeNode._Underlying *_other);
            _UnderlyingPtr = __MR_NodeNode_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NodeNode::operator=`.
        public unsafe MR.NodeNode Assign(MR.Const_NodeNode _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeNode_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NodeNode._Underlying *__MR_NodeNode_AssignFromAnother(_Underlying *_this, MR.NodeNode._Underlying *_other);
            return new(__MR_NodeNode_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NodeNode` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NodeNode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NodeNode`/`Const_NodeNode` directly.
    public class _InOptMut_NodeNode
    {
        public NodeNode? Opt;

        public _InOptMut_NodeNode() {}
        public _InOptMut_NodeNode(NodeNode value) {Opt = value;}
        public static implicit operator _InOptMut_NodeNode(NodeNode value) {return new(value);}
    }

    /// This is used for optional parameters of class `NodeNode` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NodeNode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NodeNode`/`Const_NodeNode` to pass it to the function.
    public class _InOptConst_NodeNode
    {
        public Const_NodeNode? Opt;

        public _InOptConst_NodeNode() {}
        public _InOptConst_NodeNode(Const_NodeNode value) {Opt = value;}
        public static implicit operator _InOptConst_NodeNode(Const_NodeNode value) {return new(value);}
    }
}

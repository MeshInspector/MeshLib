public static partial class MR
{
    public static partial class Std
    {
        /// Wraps a pointer to a single shared reference-counted heap-allocated `const void`.
        /// This is the const half of the class.
        public class Const_SharedPtr_ConstVoid : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_SharedPtr_ConstVoid(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_Destroy", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_Destroy(_Underlying *_this);
                __MR_std_shared_ptr_const_void_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_SharedPtr_ConstVoid() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_SharedPtr_ConstVoid() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_DefaultConstruct();
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_ConstVoid other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the stored pointer, possibly null.
            /// Returns a read-only pointer.
            public unsafe void *Get()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_Get", ExactSpelling = true)]
                extern static void *__MR_std_shared_ptr_const_void_Get(_Underlying *_this);
                return __MR_std_shared_ptr_const_void_Get(_UnderlyingPtr);
            }

            /// How many shared pointers share the managed object. Zero if no object is being managed.
            /// This being zero usually conincides with `MR_std_shared_ptr_const_void_Get()` returning null, but is ultimately orthogonal.
            /// Note that in multithreaded environments, the only safe way to use this number is comparing it with zero. Positive values might change by the time you get to use them.
            public unsafe int UseCount()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_const_void_UseCount(_Underlying *_this);
                return __MR_std_shared_ptr_const_void_UseCount(_UnderlyingPtr);
            }

            /// Create a new instance, storing a non-owning pointer.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe Const_SharedPtr_ConstVoid(void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructNonOwning", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructNonOwning(void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructNonOwning(ptr);
            }

            /// Create a new instance from a non-const pointer to the same type.
            public unsafe Const_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFromMutable", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFromMutable(MR.Misc._PassBy ptr_pass_by, MR.Std.SharedPtr_Void._Underlying *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFromMutable(ptr.PassByMode, ptr.Value is not null ? ptr.Value._UnderlyingPtr : null);
            }

            /// Create a new instance from a non-const pointer to the same type.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) {return new(ptr);}

            /// The aliasing constructor. Create a new instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructAliasing", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructAliasing(ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR._ByValue_Object _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object(MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR._ByValue_Object _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR._ByValue_Mesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh(MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR._ByValue_Mesh _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3(MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud(MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) {return new(_other);}
        }

        /// Wraps a pointer to a single shared reference-counted heap-allocated `const void`.
        /// This is the non-const half of the class.
        public class SharedPtr_ConstVoid : Const_SharedPtr_ConstVoid
        {
            internal unsafe SharedPtr_ConstVoid(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe SharedPtr_ConstVoid() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_DefaultConstruct();
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_ConstVoid other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_SharedPtr_ConstVoid other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *other);
                __MR_std_shared_ptr_const_void_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Create a new instance, storing a non-owning pointer.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe SharedPtr_ConstVoid(void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructNonOwning", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructNonOwning(void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructNonOwning(ptr);
            }

            /// Overwrite the existing instance with a non-owning pointer. The previously owned object, if any, has its reference count decremented.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe void Assign(void *ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignNonOwning", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignNonOwning(_Underlying *_this, void *ptr);
                __MR_std_shared_ptr_const_void_AssignNonOwning(_UnderlyingPtr, ptr);
            }

            /// Create a new instance from a non-const pointer to the same type.
            public unsafe SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFromMutable", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFromMutable(MR.Misc._PassBy ptr_pass_by, MR.Std.SharedPtr_Void._Underlying *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFromMutable(ptr.PassByMode, ptr.Value is not null ? ptr.Value._UnderlyingPtr : null);
            }

            /// Create a new instance from a non-const pointer to the same type.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) {return new(ptr);}

            /// Overwrite the existing instance with a non-const pointer to the same type.
            public unsafe void Assign(MR.Std._ByValue_SharedPtr_Void ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFromMutable", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFromMutable(_Underlying *_this, MR.Misc._PassBy ptr_pass_by, MR.Std.SharedPtr_Void._Underlying *ptr);
                __MR_std_shared_ptr_const_void_AssignFromMutable(_UnderlyingPtr, ptr.PassByMode, ptr.Value is not null ? ptr.Value._UnderlyingPtr : null);
            }

            /// The aliasing constructor. Create a new instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructAliasing", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructAliasing(ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// The aliasing assignment. Overwrite an existing instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a read-only pointer.
            public unsafe void AssignAliasing(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignAliasing", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignAliasing(_Underlying *_this, MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                __MR_std_shared_ptr_const_void_AssignAliasing(_UnderlyingPtr, ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_ConstVoid(MR._ByValue_Object _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object(MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Object(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR._ByValue_Object _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Object _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Object", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Object(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Object(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_ConstVoid(MR._ByValue_Mesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh(MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Mesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR._ByValue_Mesh _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Mesh _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Mesh", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Mesh(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Mesh(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3(MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_Polyline3(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Polyline3 _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Polyline3", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Polyline3(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_Polyline3(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud(MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_PointCloud(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PointCloud _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_PointCloud", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_PointCloud(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_PointCloud(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_ConstVoid._Underlying *__MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_const_void_ConstructFrom_MR_std_shared_ptr_const_MR_SceneRootObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_SceneRootObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_SceneRootObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_SceneRootObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_const_void_AssignFrom_MR_std_shared_ptr_const_MR_SceneRootObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `SharedPtr_ConstVoid` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `SharedPtr_ConstVoid`/`Const_SharedPtr_ConstVoid` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_SharedPtr_ConstVoid
        {
            internal readonly Const_SharedPtr_ConstVoid? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_SharedPtr_ConstVoid() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_SharedPtr_ConstVoid(Const_SharedPtr_ConstVoid new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_SharedPtr_ConstVoid(Const_SharedPtr_ConstVoid arg) {return new(arg);}
            public _ByValue_SharedPtr_ConstVoid(MR.Misc._Moved<SharedPtr_ConstVoid> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_SharedPtr_ConstVoid(MR.Misc._Moved<SharedPtr_ConstVoid> arg) {return new(arg);}

            /// Create a new instance from a non-const pointer to the same type.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) {return new MR.Std.SharedPtr_ConstVoid(ptr);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR._ByValue_Object _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR._ByValue_Mesh _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}
        }

        /// This is used for optional parameters of class `SharedPtr_ConstVoid` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_SharedPtr_ConstVoid`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SharedPtr_ConstVoid`/`Const_SharedPtr_ConstVoid` directly.
        public class _InOptMut_SharedPtr_ConstVoid
        {
            public SharedPtr_ConstVoid? Opt;

            public _InOptMut_SharedPtr_ConstVoid() {}
            public _InOptMut_SharedPtr_ConstVoid(SharedPtr_ConstVoid value) {Opt = value;}
            public static implicit operator _InOptMut_SharedPtr_ConstVoid(SharedPtr_ConstVoid value) {return new(value);}
        }

        /// This is used for optional parameters of class `SharedPtr_ConstVoid` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_SharedPtr_ConstVoid`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SharedPtr_ConstVoid`/`Const_SharedPtr_ConstVoid` to pass it to the function.
        public class _InOptConst_SharedPtr_ConstVoid
        {
            public Const_SharedPtr_ConstVoid? Opt;

            public _InOptConst_SharedPtr_ConstVoid() {}
            public _InOptConst_SharedPtr_ConstVoid(Const_SharedPtr_ConstVoid value) {Opt = value;}
            public static implicit operator _InOptConst_SharedPtr_ConstVoid(Const_SharedPtr_ConstVoid value) {return new(value);}

            /// Create a new instance from a non-const pointer to the same type.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR.Std._ByValue_SharedPtr_Void ptr) {return new MR.Std.SharedPtr_ConstVoid(ptr);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR._ByValue_Object _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR._ByValue_Mesh _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR._ByValue_Polyline3 _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR._ByValue_PointCloud _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_ConstVoid(MR._ByValue_SceneRootObject _other) {return new MR.Std.SharedPtr_ConstVoid(_other);}
        }
    }
}

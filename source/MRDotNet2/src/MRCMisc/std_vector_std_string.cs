public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::vector<std::string>`.
        /// This is the const half of the class.
        public class Const_Vector_StdString : MR.Misc.SharedObject, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.
            internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

            internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
            internal unsafe _Underlying *_UnderlyingPtr
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_Get", ExactSpelling = true)]
                    extern static _Underlying *__MR_std_shared_ptr_std_vector_std_string_Get(_UnderlyingShared *_this);
                    return __MR_std_shared_ptr_std_vector_std_string_Get(_UnderlyingSharedPtr);
                }
            }

            /// Check if the underlying shared pointer is owning or not.
            public override bool _IsOwning
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_UseCount", ExactSpelling = true)]
                    extern static int __MR_std_shared_ptr_std_vector_std_string_UseCount();
                    return __MR_std_shared_ptr_std_vector_std_string_UseCount() > 0;
                }
            }

            /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
            internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static _UnderlyingShared *__MR_std_shared_ptr_std_vector_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
                return __MR_std_shared_ptr_std_vector_std_string_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
            }

            internal unsafe Const_Vector_StdString(_Underlying *ptr, bool is_owning) : base(true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_Construct", ExactSpelling = true)]
                extern static _UnderlyingShared *__MR_std_shared_ptr_std_vector_std_string_Construct(_Underlying *other);
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_ConstructNonOwning", ExactSpelling = true)]
                extern static _UnderlyingShared *__MR_std_shared_ptr_std_vector_std_string_ConstructNonOwning(_Underlying *other);
                if (is_owning)
                    _UnderlyingSharedPtr = __MR_std_shared_ptr_std_vector_std_string_Construct(ptr);
                else
                    _UnderlyingSharedPtr = __MR_std_shared_ptr_std_vector_std_string_ConstructNonOwning(ptr);
            }

            internal unsafe Const_Vector_StdString(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

            internal static unsafe Vector_StdString _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_ConstructAliasing", ExactSpelling = true)]
                extern static _UnderlyingShared *__MR_std_shared_ptr_std_vector_std_string_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
                return new(__MR_std_shared_ptr_std_vector_std_string_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
            }

            private protected unsafe void _LateMakeShared(_Underlying *ptr)
            {
                System.Diagnostics.Trace.Assert(_IsOwningVal == true);
                System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_Construct", ExactSpelling = true)]
                extern static _UnderlyingShared *__MR_std_shared_ptr_std_vector_std_string_Construct(_Underlying *other);
                _UnderlyingSharedPtr = __MR_std_shared_ptr_std_vector_std_string_Construct(ptr);
            }

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_std_vector_std_string_Destroy", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_std_vector_std_string_Destroy(_UnderlyingShared *_this);
                __MR_std_shared_ptr_std_vector_std_string_Destroy(_UnderlyingSharedPtr);
                _UnderlyingSharedPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Vector_StdString() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Vector_StdString() : this(shared_ptr: null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_std_vector_std_string_DefaultConstruct();
                _LateMakeShared(__MR_std_vector_std_string_DefaultConstruct());
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Vector_StdString(MR.Std._ByValue_Vector_StdString other) : this(shared_ptr: null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_std_vector_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdString._Underlying *other);
                _LateMakeShared(__MR_std_vector_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null));
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Size", ExactSpelling = true)]
                extern static ulong __MR_std_vector_std_string_Size(_Underlying *_this);
                return __MR_std_vector_std_string_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_string_IsEmpty(_Underlying *_this);
                return __MR_std_vector_std_string_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// The memory capacity, measued in the number of elements.
            public unsafe ulong Capacity()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Capacity", ExactSpelling = true)]
                extern static ulong __MR_std_vector_std_string_Capacity(_Underlying *_this);
                return __MR_std_vector_std_string_Capacity(_UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Std.Const_String At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_At", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_std_vector_std_string_At(_Underlying *_this, ulong i);
                return new(__MR_std_vector_std_string_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, read-only.
            public unsafe MR.Std.Const_String? Front()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Front", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_std_vector_std_string_Front(_Underlying *_this);
                var __ret = __MR_std_vector_std_string_Front(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_String(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, read-only.
            public unsafe MR.Std.Const_String? Back()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Back", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_std_vector_std_string_Back(_Underlying *_this);
                var __ret = __MR_std_vector_std_string_Back(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_String(__ret, is_owning: false) : null;
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Vector_StdString.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Begin", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_Begin(_Underlying *_this);
                return new(__MR_std_vector_std_string_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Vector_StdString.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_string_IsBegin(_Underlying *_this, MR.Std.Vector_StdString.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_std_string_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Vector_StdString.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_End", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_End(_Underlying *_this);
                return new(__MR_std_vector_std_string_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Vector_StdString.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_string_IsEnd(_Underlying *_this, MR.Std.Vector_StdString.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_std_string_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Convert a const iterator to an index.
            public unsafe long ToIndex(MR.Std.Vector_StdString.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_ToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_std_string_ToIndex(_Underlying *_this, MR.Std.Vector_StdString.ConstIterator._Underlying *iter);
                return __MR_std_vector_std_string_ToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Convert a mutable iterator to an index.
            public unsafe long MutableToIndex(MR.Std.Vector_StdString.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_std_string_MutableToIndex(_Underlying *_this, MR.Std.Vector_StdString.Iterator._Underlying *iter);
                return __MR_std_vector_std_string_MutableToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Read-only iterator for `MR_std_vector_std_string`.
            /// This is the const half of the class.
            public class Const_ConstIterator : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_ConstIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_std_string_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Vector_StdString.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_ConstructFromAnother(MR.Std.Vector_StdString.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Std.Vector_StdString.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_FromMutable(MR.Std.Vector_StdString.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Std.Vector_StdString.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator.
                public unsafe MR.Std.Const_String Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_std_vector_std_string_const_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_std_string_const_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two const iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_StdString.Const_ConstIterator a, MR.Std.Vector_StdString.Const_ConstIterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_std_string_const_iterator_Distance(MR.Std.Vector_StdString.ConstIterator._Underlying *a, MR.Std.Vector_StdString.ConstIterator._Underlying *b);
                    return __MR_std_vector_std_string_const_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Read-only iterator for `MR_std_vector_std_string`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Vector_StdString.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_ConstructFromAnother(MR.Std.Vector_StdString.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_StdString.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_StdString.ConstIterator._Underlying *other);
                    __MR_std_vector_std_string_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Std.Vector_StdString.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.ConstIterator._Underlying *__MR_std_vector_std_string_const_iterator_FromMutable(MR.Std.Vector_StdString.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_std_string_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Std.Vector_StdString.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_const_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_std_string_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_const_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_std_string_const_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a const iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_const_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_const_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_std_string_const_iterator_OffsetBy(_UnderlyingPtr, delta);
                }
            }

            /// This is used for optional parameters of class `ConstIterator` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_ConstIterator`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `ConstIterator`/`Const_ConstIterator` directly.
            public class _InOptMut_ConstIterator
            {
                public ConstIterator? Opt;

                public _InOptMut_ConstIterator() {}
                public _InOptMut_ConstIterator(ConstIterator value) {Opt = value;}
                public static implicit operator _InOptMut_ConstIterator(ConstIterator value) {return new(value);}
            }

            /// This is used for optional parameters of class `ConstIterator` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_ConstIterator`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `ConstIterator`/`Const_ConstIterator` to pass it to the function.
            public class _InOptConst_ConstIterator
            {
                public Const_ConstIterator? Opt;

                public _InOptConst_ConstIterator() {}
                public _InOptConst_ConstIterator(Const_ConstIterator value) {Opt = value;}
                public static implicit operator _InOptConst_ConstIterator(Const_ConstIterator value) {return new(value);}

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Std.Vector_StdString.Const_Iterator iter) {return new MR.Std.Vector_StdString.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_std_vector_std_string`.
            /// This is the const half of the class.
            public class Const_Iterator : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Iterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_std_string_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_string_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Std.Vector_StdString.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_iterator_ConstructFromAnother(MR.Std.Vector_StdString.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_string_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator.
                public unsafe MR.Std.String Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_std_vector_std_string_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_std_string_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two mutable iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_StdString.Const_Iterator a, MR.Std.Vector_StdString.Const_Iterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_std_string_iterator_Distance(MR.Std.Vector_StdString.Iterator._Underlying *a, MR.Std.Vector_StdString.Iterator._Underlying *b);
                    return __MR_std_vector_std_string_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Mutable iterator for `MR_std_vector_std_string`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_string_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Std.Vector_StdString.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_iterator_ConstructFromAnother(MR.Std.Vector_StdString.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_string_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_StdString.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_StdString.Iterator._Underlying *other);
                    __MR_std_vector_std_string_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_std_string_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a mutable iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_std_string_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a mutable iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_string_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_std_string_iterator_OffsetBy(_UnderlyingPtr, delta);
                }
            }

            /// This is used for optional parameters of class `Iterator` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Iterator`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Iterator`/`Const_Iterator` directly.
            public class _InOptMut_Iterator
            {
                public Iterator? Opt;

                public _InOptMut_Iterator() {}
                public _InOptMut_Iterator(Iterator value) {Opt = value;}
                public static implicit operator _InOptMut_Iterator(Iterator value) {return new(value);}
            }

            /// This is used for optional parameters of class `Iterator` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Iterator`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Iterator`/`Const_Iterator` to pass it to the function.
            public class _InOptConst_Iterator
            {
                public Const_Iterator? Opt;

                public _InOptConst_Iterator() {}
                public _InOptConst_Iterator(Const_Iterator value) {Opt = value;}
                public static implicit operator _InOptConst_Iterator(Const_Iterator value) {return new(value);}
            }
        }

        /// Generated from C++ container `std::vector<std::string>`.
        /// This is the non-const half of the class.
        public class Vector_StdString : Const_Vector_StdString
        {
            internal unsafe Vector_StdString(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            internal unsafe Vector_StdString(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Vector_StdString() : this(shared_ptr: null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_std_vector_std_string_DefaultConstruct();
                _LateMakeShared(__MR_std_vector_std_string_DefaultConstruct());
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Vector_StdString(MR.Std._ByValue_Vector_StdString other) : this(shared_ptr: null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_std_vector_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdString._Underlying *other);
                _LateMakeShared(__MR_std_vector_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null));
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Vector_StdString other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdString._Underlying *other);
                __MR_std_vector_std_string_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Resizes the container. The new elements if any are zeroed.
            public unsafe void Resize(ulong new_size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Resize", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_Resize(_Underlying *_this, ulong new_size);
                __MR_std_vector_std_string_Resize(_UnderlyingPtr, new_size);
            }

            /// Resizes the container. The new elements if any are set to the specified value.
            public unsafe void ResizeWithDefaultValue(ulong new_size, ReadOnlySpan<char> value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_ResizeWithDefaultValue", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_ResizeWithDefaultValue(_Underlying *_this, ulong new_size, byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    __MR_std_vector_std_string_ResizeWithDefaultValue(_UnderlyingPtr, new_size, __ptr_value, __ptr_value + __len_value);
                }
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Clear", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_Clear(_Underlying *_this);
                __MR_std_vector_std_string_Clear(_UnderlyingPtr);
            }

            /// Reserves memory for a certain number of elements. Never shrinks the memory.
            public unsafe void Reserve(ulong new_capacity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Reserve", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_Reserve(_Underlying *_this, ulong new_capacity);
                __MR_std_vector_std_string_Reserve(_UnderlyingPtr, new_capacity);
            }

            /// Shrinks the capacity to match the size.
            public unsafe void ShrinkToFit()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_ShrinkToFit", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_ShrinkToFit(_Underlying *_this);
                __MR_std_vector_std_string_ShrinkToFit(_UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.Std.String MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableAt", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_vector_std_string_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_vector_std_string_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, mutable.
            public unsafe MR.Std.String? MutableFront()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableFront", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_vector_std_string_MutableFront(_Underlying *_this);
                var __ret = __MR_std_vector_std_string_MutableFront(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.String(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, mutable.
            public unsafe MR.Std.String? MutableBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableBack", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_vector_std_string_MutableBack(_Underlying *_this);
                var __ret = __MR_std_vector_std_string_MutableBack(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.String(__ret, is_owning: false) : null;
            }

            /// Inserts a new element at the end.
            public unsafe void PushBack(ReadOnlySpan<char> new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_PushBack", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_PushBack(_Underlying *_this, byte *new_elem, byte *new_elem_end);
                byte[] __bytes_new_elem = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(new_elem.Length)];
                int __len_new_elem = System.Text.Encoding.UTF8.GetBytes(new_elem, __bytes_new_elem);
                fixed (byte *__ptr_new_elem = __bytes_new_elem)
                {
                    __MR_std_vector_std_string_PushBack(_UnderlyingPtr, __ptr_new_elem, __ptr_new_elem + __len_new_elem);
                }
            }

            /// Removes one element from the end.
            public unsafe void PopBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_PopBack", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_PopBack(_Underlying *_this);
                __MR_std_vector_std_string_PopBack(_UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void Insert(ulong position, ReadOnlySpan<char> new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Insert", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_Insert(_Underlying *_this, ulong position, byte *new_elem, byte *new_elem_end);
                byte[] __bytes_new_elem = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(new_elem.Length)];
                int __len_new_elem = System.Text.Encoding.UTF8.GetBytes(new_elem, __bytes_new_elem);
                fixed (byte *__ptr_new_elem = __bytes_new_elem)
                {
                    __MR_std_vector_std_string_Insert(_UnderlyingPtr, position, __ptr_new_elem, __ptr_new_elem + __len_new_elem);
                }
            }

            /// Erases the element at the specified position.
            public unsafe void Erase(ulong position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_Erase", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_Erase(_Underlying *_this, ulong position);
                __MR_std_vector_std_string_Erase(_UnderlyingPtr, position);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void InsertAtMutableIter(MR.Std.Vector_StdString.Const_Iterator position, ReadOnlySpan<char> new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_InsertAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_InsertAtMutableIter(_Underlying *_this, MR.Std.Vector_StdString.Iterator._Underlying *position, byte *new_elem, byte *new_elem_end);
                byte[] __bytes_new_elem = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(new_elem.Length)];
                int __len_new_elem = System.Text.Encoding.UTF8.GetBytes(new_elem, __bytes_new_elem);
                fixed (byte *__ptr_new_elem = __bytes_new_elem)
                {
                    __MR_std_vector_std_string_InsertAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr, __ptr_new_elem, __ptr_new_elem + __len_new_elem);
                }
            }

            /// Erases the element at the specified position.
            public unsafe void EraseAtMutableIter(MR.Std.Vector_StdString.Const_Iterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_EraseAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_EraseAtMutableIter(_Underlying *_this, MR.Std.Vector_StdString.Iterator._Underlying *position);
                __MR_std_vector_std_string_EraseAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void InsertAtIter(MR.Std.Vector_StdString.Const_ConstIterator position, ReadOnlySpan<char> new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_InsertAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_InsertAtIter(_Underlying *_this, MR.Std.Vector_StdString.ConstIterator._Underlying *position, byte *new_elem, byte *new_elem_end);
                byte[] __bytes_new_elem = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(new_elem.Length)];
                int __len_new_elem = System.Text.Encoding.UTF8.GetBytes(new_elem, __bytes_new_elem);
                fixed (byte *__ptr_new_elem = __bytes_new_elem)
                {
                    __MR_std_vector_std_string_InsertAtIter(_UnderlyingPtr, position._UnderlyingPtr, __ptr_new_elem, __ptr_new_elem + __len_new_elem);
                }
            }

            /// Erases the element at the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void EraseAtIter(MR.Std.Vector_StdString.Const_ConstIterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_EraseAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_string_EraseAtIter(_Underlying *_this, MR.Std.Vector_StdString.ConstIterator._Underlying *position);
                __MR_std_vector_std_string_EraseAtIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Std.Vector_StdString.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableBegin", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_MutableBegin(_Underlying *_this);
                return new(__MR_std_vector_std_string_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Std.Vector_StdString.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_string_IsMutableBegin(_Underlying *_this, MR.Std.Vector_StdString.Const_Iterator._Underlying *iter);
                return __MR_std_vector_std_string_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Std.Vector_StdString.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_MutableEnd", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString.Iterator._Underlying *__MR_std_vector_std_string_MutableEnd(_Underlying *_this);
                return new(__MR_std_vector_std_string_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Std.Vector_StdString.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_string_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_string_IsMutableEnd(_Underlying *_this, MR.Std.Vector_StdString.Const_Iterator._Underlying *iter);
                return __MR_std_vector_std_string_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Vector_StdString` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Vector_StdString`/`Const_Vector_StdString` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Vector_StdString
        {
            internal readonly Const_Vector_StdString? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Vector_StdString() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Vector_StdString(Const_Vector_StdString new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Vector_StdString(Const_Vector_StdString arg) {return new(arg);}
            public _ByValue_Vector_StdString(MR.Misc._Moved<Vector_StdString> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Vector_StdString(MR.Misc._Moved<Vector_StdString> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Vector_StdString` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector_StdString`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_StdString`/`Const_Vector_StdString` directly.
        public class _InOptMut_Vector_StdString
        {
            public Vector_StdString? Opt;

            public _InOptMut_Vector_StdString() {}
            public _InOptMut_Vector_StdString(Vector_StdString value) {Opt = value;}
            public static implicit operator _InOptMut_Vector_StdString(Vector_StdString value) {return new(value);}
        }

        /// This is used for optional parameters of class `Vector_StdString` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector_StdString`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_StdString`/`Const_Vector_StdString` to pass it to the function.
        public class _InOptConst_Vector_StdString
        {
            public Const_Vector_StdString? Opt;

            public _InOptConst_Vector_StdString() {}
            public _InOptConst_Vector_StdString(Const_Vector_StdString value) {Opt = value;}
            public static implicit operator _InOptConst_Vector_StdString(Const_Vector_StdString value) {return new(value);}
        }
    }
}

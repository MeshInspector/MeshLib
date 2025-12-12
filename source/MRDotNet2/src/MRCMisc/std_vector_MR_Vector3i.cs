public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::vector<MR::Vector3i>`.
        /// This is the const half of the class.
        public class Const_Vector_MRVector3i : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Vector_MRVector3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Destroy", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Destroy(_Underlying *_this);
                __MR_std_vector_MR_Vector3i_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Vector_MRVector3i() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Vector_MRVector3i() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Vector_MRVector3i(MR.Std._ByValue_Vector_MRVector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRVector3i._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Construct from a range of elements.
            public unsafe Const_Vector_MRVector3i(MR.Const_Vector3i? ptr, ulong size) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ConstructFromRange", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_ConstructFromRange(MR.Const_Vector3i._Underlying *ptr, ulong size);
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_ConstructFromRange(ptr is not null ? ptr._UnderlyingPtr : null, size);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Size", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_Vector3i_Size(_Underlying *_this);
                return __MR_std_vector_MR_Vector3i_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_Vector3i_IsEmpty(_Underlying *_this);
                return __MR_std_vector_MR_Vector3i_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// The memory capacity, measued in the number of elements.
            public unsafe ulong Capacity()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Capacity", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_Vector3i_Capacity(_Underlying *_this);
                return __MR_std_vector_MR_Vector3i_Capacity(_UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Const_Vector3i At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_At", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_At(_Underlying *_this, ulong i);
                return new(__MR_std_vector_MR_Vector3i_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, read-only.
            public unsafe MR.Const_Vector3i? Front()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Front", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_Front(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_Front(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Vector3i(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, read-only.
            public unsafe MR.Const_Vector3i? Back()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Back", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_Back(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_Back(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Vector3i(__ret, is_owning: false) : null;
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            public unsafe MR.Const_Vector3i? Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Data", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_Data(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_Data(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Vector3i(__ret, is_owning: false) : null;
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Vector_MRVector3i.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Begin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_Begin(_Underlying *_this);
                return new(__MR_std_vector_MR_Vector3i_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Vector_MRVector3i.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_Vector3i_IsBegin(_Underlying *_this, MR.Std.Vector_MRVector3i.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Vector_MRVector3i.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_End", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_End(_Underlying *_this);
                return new(__MR_std_vector_MR_Vector3i_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Vector_MRVector3i.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_Vector3i_IsEnd(_Underlying *_this, MR.Std.Vector_MRVector3i.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Convert a const iterator to an index.
            public unsafe long ToIndex(MR.Std.Vector_MRVector3i.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_Vector3i_ToIndex(_Underlying *_this, MR.Std.Vector_MRVector3i.ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_ToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Convert a mutable iterator to an index.
            public unsafe long MutableToIndex(MR.Std.Vector_MRVector3i.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_Vector3i_MutableToIndex(_Underlying *_this, MR.Std.Vector_MRVector3i.Iterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_MutableToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Read-only iterator for `MR_std_vector_MR_Vector3i`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRVector3i.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother(MR.Std.Vector_MRVector3i.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRVector3i.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_FromMutable(MR.Std.Vector_MRVector3i.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Std.Vector_MRVector3i.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator.
                public unsafe MR.Const_Vector3i Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Const_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_MR_Vector3i_const_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two const iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRVector3i.Const_ConstIterator a, MR.Std.Vector_MRVector3i.Const_ConstIterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_Vector3i_const_iterator_Distance(MR.Std.Vector_MRVector3i.ConstIterator._Underlying *a, MR.Std.Vector_MRVector3i.ConstIterator._Underlying *b);
                    return __MR_std_vector_MR_Vector3i_const_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Read-only iterator for `MR_std_vector_MR_Vector3i`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Vector_MRVector3i.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother(MR.Std.Vector_MRVector3i.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRVector3i.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRVector3i.ConstIterator._Underlying *other);
                    __MR_std_vector_MR_Vector3i_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Std.Vector_MRVector3i.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.ConstIterator._Underlying *__MR_std_vector_MR_Vector3i_const_iterator_FromMutable(MR.Std.Vector_MRVector3i.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Std.Vector_MRVector3i.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_const_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_const_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_const_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a const iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_const_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_const_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_Vector3i_const_iterator_OffsetBy(_UnderlyingPtr, delta);
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
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Std.Vector_MRVector3i.Const_Iterator iter) {return new MR.Std.Vector_MRVector3i.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_std_vector_MR_Vector3i`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Std.Vector_MRVector3i.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother(MR.Std.Vector_MRVector3i.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator.
                public unsafe MR.Mut_Vector3i Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Mut_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_MR_Vector3i_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two mutable iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRVector3i.Const_Iterator a, MR.Std.Vector_MRVector3i.Const_Iterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_Vector3i_iterator_Distance(MR.Std.Vector_MRVector3i.Iterator._Underlying *a, MR.Std.Vector_MRVector3i.Iterator._Underlying *b);
                    return __MR_std_vector_MR_Vector3i_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Mutable iterator for `MR_std_vector_MR_Vector3i`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Std.Vector_MRVector3i.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother(MR.Std.Vector_MRVector3i.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_Vector3i_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRVector3i.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRVector3i.Iterator._Underlying *other);
                    __MR_std_vector_MR_Vector3i_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a mutable iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_Vector3i_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a mutable iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_Vector3i_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_Vector3i_iterator_OffsetBy(_UnderlyingPtr, delta);
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

        /// Generated from C++ container `std::vector<MR::Vector3i>`.
        /// This is the non-const half of the class.
        public class Vector_MRVector3i : Const_Vector_MRVector3i
        {
            internal unsafe Vector_MRVector3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Vector_MRVector3i() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Vector_MRVector3i(MR.Std._ByValue_Vector_MRVector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRVector3i._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Vector_MRVector3i other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRVector3i._Underlying *other);
                __MR_std_vector_MR_Vector3i_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Construct from a range of elements.
            public unsafe Vector_MRVector3i(MR.Const_Vector3i? ptr, ulong size) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ConstructFromRange", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i._Underlying *__MR_std_vector_MR_Vector3i_ConstructFromRange(MR.Const_Vector3i._Underlying *ptr, ulong size);
                _UnderlyingPtr = __MR_std_vector_MR_Vector3i_ConstructFromRange(ptr is not null ? ptr._UnderlyingPtr : null, size);
            }

            /// Assign from a range of elements, overwriting previous contents.
            public unsafe void Assign(MR.Const_Vector3i? ptr, ulong size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_AssignFromRange", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_AssignFromRange(_Underlying *_this, MR.Const_Vector3i._Underlying *ptr, ulong size);
                __MR_std_vector_MR_Vector3i_AssignFromRange(_UnderlyingPtr, ptr is not null ? ptr._UnderlyingPtr : null, size);
            }

            /// Resizes the container. The new elements if any are zeroed.
            public unsafe void Resize(ulong new_size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Resize", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Resize(_Underlying *_this, ulong new_size);
                __MR_std_vector_MR_Vector3i_Resize(_UnderlyingPtr, new_size);
            }

            /// Resizes the container. The new elements if any are set to the specified value.
            public unsafe void ResizeWithDefaultValue(ulong new_size, MR.Const_Vector3i value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ResizeWithDefaultValue", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_ResizeWithDefaultValue(_Underlying *_this, ulong new_size, MR.Const_Vector3i._Underlying *value);
                __MR_std_vector_MR_Vector3i_ResizeWithDefaultValue(_UnderlyingPtr, new_size, value._UnderlyingPtr);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Clear", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Clear(_Underlying *_this);
                __MR_std_vector_MR_Vector3i_Clear(_UnderlyingPtr);
            }

            /// Reserves memory for a certain number of elements. Never shrinks the memory.
            public unsafe void Reserve(ulong new_capacity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Reserve", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Reserve(_Underlying *_this, ulong new_capacity);
                __MR_std_vector_MR_Vector3i_Reserve(_UnderlyingPtr, new_capacity);
            }

            /// Shrinks the capacity to match the size.
            public unsafe void ShrinkToFit()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_ShrinkToFit", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_ShrinkToFit(_Underlying *_this);
                __MR_std_vector_MR_Vector3i_ShrinkToFit(_UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.Mut_Vector3i MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableAt", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_vector_MR_Vector3i_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, mutable.
            public unsafe MR.Mut_Vector3i? MutableFront()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableFront", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_MutableFront(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_MutableFront(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Vector3i(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, mutable.
            public unsafe MR.Mut_Vector3i? MutableBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableBack", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_MutableBack(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_MutableBack(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Vector3i(__ret, is_owning: false) : null;
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            public unsafe MR.Mut_Vector3i? MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableData", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_std_vector_MR_Vector3i_MutableData(_Underlying *_this);
                var __ret = __MR_std_vector_MR_Vector3i_MutableData(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Vector3i(__ret, is_owning: false) : null;
            }

            /// Inserts a new element at the end.
            public unsafe void PushBack(MR.Vector3i new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_PushBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_PushBack(_Underlying *_this, MR.Vector3i new_elem);
                __MR_std_vector_MR_Vector3i_PushBack(_UnderlyingPtr, new_elem);
            }

            /// Removes one element from the end.
            public unsafe void PopBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_PopBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_PopBack(_Underlying *_this);
                __MR_std_vector_MR_Vector3i_PopBack(_UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void Insert(ulong position, MR.Vector3i new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Insert", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Insert(_Underlying *_this, ulong position, MR.Vector3i new_elem);
                __MR_std_vector_MR_Vector3i_Insert(_UnderlyingPtr, position, new_elem);
            }

            /// Erases the element at the specified position.
            public unsafe void Erase(ulong position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_Erase", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_Erase(_Underlying *_this, ulong position);
                __MR_std_vector_MR_Vector3i_Erase(_UnderlyingPtr, position);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void InsertAtMutableIter(MR.Std.Vector_MRVector3i.Const_Iterator position, MR.Vector3i new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_InsertAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_InsertAtMutableIter(_Underlying *_this, MR.Std.Vector_MRVector3i.Iterator._Underlying *position, MR.Vector3i new_elem);
                __MR_std_vector_MR_Vector3i_InsertAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem);
            }

            /// Erases the element at the specified position.
            public unsafe void EraseAtMutableIter(MR.Std.Vector_MRVector3i.Const_Iterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_EraseAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_EraseAtMutableIter(_Underlying *_this, MR.Std.Vector_MRVector3i.Iterator._Underlying *position);
                __MR_std_vector_MR_Vector3i_EraseAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void InsertAtIter(MR.Std.Vector_MRVector3i.Const_ConstIterator position, MR.Vector3i new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_InsertAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_InsertAtIter(_Underlying *_this, MR.Std.Vector_MRVector3i.ConstIterator._Underlying *position, MR.Vector3i new_elem);
                __MR_std_vector_MR_Vector3i_InsertAtIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem);
            }

            /// Erases the element at the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void EraseAtIter(MR.Std.Vector_MRVector3i.Const_ConstIterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_EraseAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_Vector3i_EraseAtIter(_Underlying *_this, MR.Std.Vector_MRVector3i.ConstIterator._Underlying *position);
                __MR_std_vector_MR_Vector3i_EraseAtIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Std.Vector_MRVector3i.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableBegin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_MutableBegin(_Underlying *_this);
                return new(__MR_std_vector_MR_Vector3i_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Std.Vector_MRVector3i.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_Vector3i_IsMutableBegin(_Underlying *_this, MR.Std.Vector_MRVector3i.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Std.Vector_MRVector3i.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_MutableEnd", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3i.Iterator._Underlying *__MR_std_vector_MR_Vector3i_MutableEnd(_Underlying *_this);
                return new(__MR_std_vector_MR_Vector3i_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Std.Vector_MRVector3i.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_Vector3i_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_Vector3i_IsMutableEnd(_Underlying *_this, MR.Std.Vector_MRVector3i.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_Vector3i_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Vector_MRVector3i` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Vector_MRVector3i`/`Const_Vector_MRVector3i` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Vector_MRVector3i
        {
            internal readonly Const_Vector_MRVector3i? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Vector_MRVector3i() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Vector_MRVector3i(Const_Vector_MRVector3i new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Vector_MRVector3i(Const_Vector_MRVector3i arg) {return new(arg);}
            public _ByValue_Vector_MRVector3i(MR.Misc._Moved<Vector_MRVector3i> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Vector_MRVector3i(MR.Misc._Moved<Vector_MRVector3i> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Vector_MRVector3i` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector_MRVector3i`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRVector3i`/`Const_Vector_MRVector3i` directly.
        public class _InOptMut_Vector_MRVector3i
        {
            public Vector_MRVector3i? Opt;

            public _InOptMut_Vector_MRVector3i() {}
            public _InOptMut_Vector_MRVector3i(Vector_MRVector3i value) {Opt = value;}
            public static implicit operator _InOptMut_Vector_MRVector3i(Vector_MRVector3i value) {return new(value);}
        }

        /// This is used for optional parameters of class `Vector_MRVector3i` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector_MRVector3i`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRVector3i`/`Const_Vector_MRVector3i` to pass it to the function.
        public class _InOptConst_Vector_MRVector3i
        {
            public Const_Vector_MRVector3i? Opt;

            public _InOptConst_Vector_MRVector3i() {}
            public _InOptConst_Vector_MRVector3i(Const_Vector_MRVector3i value) {Opt = value;}
            public static implicit operator _InOptConst_Vector_MRVector3i(Const_Vector_MRVector3i value) {return new(value);}
        }
    }
}

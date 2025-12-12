public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::vector<MR::SkyPatch>`.
        /// This is the const half of the class.
        public class Const_Vector_MRSkyPatch : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Vector_MRSkyPatch(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Destroy", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Destroy(_Underlying *_this);
                __MR_std_vector_MR_SkyPatch_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Vector_MRSkyPatch() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Vector_MRSkyPatch() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Vector_MRSkyPatch(MR.Std._ByValue_Vector_MRSkyPatch other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSkyPatch._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Size", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_SkyPatch_Size(_Underlying *_this);
                return __MR_std_vector_MR_SkyPatch_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_SkyPatch_IsEmpty(_Underlying *_this);
                return __MR_std_vector_MR_SkyPatch_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// The memory capacity, measued in the number of elements.
            public unsafe ulong Capacity()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Capacity", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_SkyPatch_Capacity(_Underlying *_this);
                return __MR_std_vector_MR_SkyPatch_Capacity(_UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Const_SkyPatch At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_At", ExactSpelling = true)]
                extern static MR.Const_SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_At(_Underlying *_this, ulong i);
                return new(__MR_std_vector_MR_SkyPatch_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, read-only.
            public unsafe MR.Const_SkyPatch? Front()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Front", ExactSpelling = true)]
                extern static MR.Const_SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_Front(_Underlying *_this);
                var __ret = __MR_std_vector_MR_SkyPatch_Front(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_SkyPatch(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, read-only.
            public unsafe MR.Const_SkyPatch? Back()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Back", ExactSpelling = true)]
                extern static MR.Const_SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_Back(_Underlying *_this);
                var __ret = __MR_std_vector_MR_SkyPatch_Back(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_SkyPatch(__ret, is_owning: false) : null;
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Vector_MRSkyPatch.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Begin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_Begin(_Underlying *_this);
                return new(__MR_std_vector_MR_SkyPatch_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Vector_MRSkyPatch.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_SkyPatch_IsBegin(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Vector_MRSkyPatch.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_End", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_End(_Underlying *_this);
                return new(__MR_std_vector_MR_SkyPatch_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Vector_MRSkyPatch.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_SkyPatch_IsEnd(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Convert a const iterator to an index.
            public unsafe long ToIndex(MR.Std.Vector_MRSkyPatch.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_ToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_SkyPatch_ToIndex(_Underlying *_this, MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_ToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Convert a mutable iterator to an index.
            public unsafe long MutableToIndex(MR.Std.Vector_MRSkyPatch.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_SkyPatch_MutableToIndex(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Iterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_MutableToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Read-only iterator for `MR_std_vector_MR_SkyPatch`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRSkyPatch.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother(MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRSkyPatch.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_FromMutable(MR.Std.Vector_MRSkyPatch.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Std.Vector_MRSkyPatch.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator.
                public unsafe MR.Const_SkyPatch Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Const_SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_MR_SkyPatch_const_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two const iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRSkyPatch.Const_ConstIterator a, MR.Std.Vector_MRSkyPatch.Const_ConstIterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_SkyPatch_const_iterator_Distance(MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *a, MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *b);
                    return __MR_std_vector_MR_SkyPatch_const_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Read-only iterator for `MR_std_vector_MR_SkyPatch`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Vector_MRSkyPatch.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother(MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRSkyPatch.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *other);
                    __MR_std_vector_MR_SkyPatch_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Std.Vector_MRSkyPatch.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *__MR_std_vector_MR_SkyPatch_const_iterator_FromMutable(MR.Std.Vector_MRSkyPatch.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Std.Vector_MRSkyPatch.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_const_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_const_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_const_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a const iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_const_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_const_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_SkyPatch_const_iterator_OffsetBy(_UnderlyingPtr, delta);
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
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Std.Vector_MRSkyPatch.Const_Iterator iter) {return new MR.Std.Vector_MRSkyPatch.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_std_vector_MR_SkyPatch`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Std.Vector_MRSkyPatch.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother(MR.Std.Vector_MRSkyPatch.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator.
                public unsafe MR.SkyPatch Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_Deref", ExactSpelling = true)]
                    extern static MR.SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_MR_SkyPatch_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two mutable iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRSkyPatch.Const_Iterator a, MR.Std.Vector_MRSkyPatch.Const_Iterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_SkyPatch_iterator_Distance(MR.Std.Vector_MRSkyPatch.Iterator._Underlying *a, MR.Std.Vector_MRSkyPatch.Iterator._Underlying *b);
                    return __MR_std_vector_MR_SkyPatch_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Mutable iterator for `MR_std_vector_MR_SkyPatch`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Std.Vector_MRSkyPatch.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother(MR.Std.Vector_MRSkyPatch.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRSkyPatch.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Iterator._Underlying *other);
                    __MR_std_vector_MR_SkyPatch_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a mutable iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_SkyPatch_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a mutable iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_SkyPatch_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_SkyPatch_iterator_OffsetBy(_UnderlyingPtr, delta);
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

        /// Generated from C++ container `std::vector<MR::SkyPatch>`.
        /// This is the non-const half of the class.
        public class Vector_MRSkyPatch : Const_Vector_MRSkyPatch
        {
            internal unsafe Vector_MRSkyPatch(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Vector_MRSkyPatch() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Vector_MRSkyPatch(MR.Std._ByValue_Vector_MRSkyPatch other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSkyPatch._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_SkyPatch_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Vector_MRSkyPatch other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSkyPatch._Underlying *other);
                __MR_std_vector_MR_SkyPatch_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Resizes the container. The new elements if any are zeroed.
            public unsafe void Resize(ulong new_size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Resize", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Resize(_Underlying *_this, ulong new_size);
                __MR_std_vector_MR_SkyPatch_Resize(_UnderlyingPtr, new_size);
            }

            /// Resizes the container. The new elements if any are set to the specified value.
            public unsafe void ResizeWithDefaultValue(ulong new_size, MR.Const_SkyPatch value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_ResizeWithDefaultValue", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_ResizeWithDefaultValue(_Underlying *_this, ulong new_size, MR.Const_SkyPatch._Underlying *value);
                __MR_std_vector_MR_SkyPatch_ResizeWithDefaultValue(_UnderlyingPtr, new_size, value._UnderlyingPtr);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Clear", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Clear(_Underlying *_this);
                __MR_std_vector_MR_SkyPatch_Clear(_UnderlyingPtr);
            }

            /// Reserves memory for a certain number of elements. Never shrinks the memory.
            public unsafe void Reserve(ulong new_capacity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Reserve", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Reserve(_Underlying *_this, ulong new_capacity);
                __MR_std_vector_MR_SkyPatch_Reserve(_UnderlyingPtr, new_capacity);
            }

            /// Shrinks the capacity to match the size.
            public unsafe void ShrinkToFit()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_ShrinkToFit", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_ShrinkToFit(_Underlying *_this);
                __MR_std_vector_MR_SkyPatch_ShrinkToFit(_UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.SkyPatch MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableAt", ExactSpelling = true)]
                extern static MR.SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_vector_MR_SkyPatch_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, mutable.
            public unsafe MR.SkyPatch? MutableFront()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableFront", ExactSpelling = true)]
                extern static MR.SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_MutableFront(_Underlying *_this);
                var __ret = __MR_std_vector_MR_SkyPatch_MutableFront(_UnderlyingPtr);
                return __ret is not null ? new MR.SkyPatch(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, mutable.
            public unsafe MR.SkyPatch? MutableBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableBack", ExactSpelling = true)]
                extern static MR.SkyPatch._Underlying *__MR_std_vector_MR_SkyPatch_MutableBack(_Underlying *_this);
                var __ret = __MR_std_vector_MR_SkyPatch_MutableBack(_UnderlyingPtr);
                return __ret is not null ? new MR.SkyPatch(__ret, is_owning: false) : null;
            }

            /// Inserts a new element at the end.
            public unsafe void PushBack(MR.Const_SkyPatch new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_PushBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_PushBack(_Underlying *_this, MR.SkyPatch._Underlying *new_elem);
                __MR_std_vector_MR_SkyPatch_PushBack(_UnderlyingPtr, new_elem._UnderlyingPtr);
            }

            /// Removes one element from the end.
            public unsafe void PopBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_PopBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_PopBack(_Underlying *_this);
                __MR_std_vector_MR_SkyPatch_PopBack(_UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void Insert(ulong position, MR.Const_SkyPatch new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Insert", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Insert(_Underlying *_this, ulong position, MR.SkyPatch._Underlying *new_elem);
                __MR_std_vector_MR_SkyPatch_Insert(_UnderlyingPtr, position, new_elem._UnderlyingPtr);
            }

            /// Erases the element at the specified position.
            public unsafe void Erase(ulong position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_Erase", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_Erase(_Underlying *_this, ulong position);
                __MR_std_vector_MR_SkyPatch_Erase(_UnderlyingPtr, position);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void InsertAtMutableIter(MR.Std.Vector_MRSkyPatch.Const_Iterator position, MR.Const_SkyPatch new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_InsertAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_InsertAtMutableIter(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Iterator._Underlying *position, MR.SkyPatch._Underlying *new_elem);
                __MR_std_vector_MR_SkyPatch_InsertAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem._UnderlyingPtr);
            }

            /// Erases the element at the specified position.
            public unsafe void EraseAtMutableIter(MR.Std.Vector_MRSkyPatch.Const_Iterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_EraseAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_EraseAtMutableIter(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Iterator._Underlying *position);
                __MR_std_vector_MR_SkyPatch_EraseAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void InsertAtIter(MR.Std.Vector_MRSkyPatch.Const_ConstIterator position, MR.Const_SkyPatch new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_InsertAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_InsertAtIter(_Underlying *_this, MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *position, MR.SkyPatch._Underlying *new_elem);
                __MR_std_vector_MR_SkyPatch_InsertAtIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem._UnderlyingPtr);
            }

            /// Erases the element at the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void EraseAtIter(MR.Std.Vector_MRSkyPatch.Const_ConstIterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_EraseAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_SkyPatch_EraseAtIter(_Underlying *_this, MR.Std.Vector_MRSkyPatch.ConstIterator._Underlying *position);
                __MR_std_vector_MR_SkyPatch_EraseAtIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Std.Vector_MRSkyPatch.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableBegin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_MutableBegin(_Underlying *_this);
                return new(__MR_std_vector_MR_SkyPatch_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Std.Vector_MRSkyPatch.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_SkyPatch_IsMutableBegin(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Std.Vector_MRSkyPatch.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_MutableEnd", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSkyPatch.Iterator._Underlying *__MR_std_vector_MR_SkyPatch_MutableEnd(_Underlying *_this);
                return new(__MR_std_vector_MR_SkyPatch_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Std.Vector_MRSkyPatch.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_SkyPatch_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_SkyPatch_IsMutableEnd(_Underlying *_this, MR.Std.Vector_MRSkyPatch.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_SkyPatch_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Vector_MRSkyPatch` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Vector_MRSkyPatch`/`Const_Vector_MRSkyPatch` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Vector_MRSkyPatch
        {
            internal readonly Const_Vector_MRSkyPatch? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Vector_MRSkyPatch() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Vector_MRSkyPatch(Const_Vector_MRSkyPatch new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Vector_MRSkyPatch(Const_Vector_MRSkyPatch arg) {return new(arg);}
            public _ByValue_Vector_MRSkyPatch(MR.Misc._Moved<Vector_MRSkyPatch> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Vector_MRSkyPatch(MR.Misc._Moved<Vector_MRSkyPatch> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Vector_MRSkyPatch` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector_MRSkyPatch`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRSkyPatch`/`Const_Vector_MRSkyPatch` directly.
        public class _InOptMut_Vector_MRSkyPatch
        {
            public Vector_MRSkyPatch? Opt;

            public _InOptMut_Vector_MRSkyPatch() {}
            public _InOptMut_Vector_MRSkyPatch(Vector_MRSkyPatch value) {Opt = value;}
            public static implicit operator _InOptMut_Vector_MRSkyPatch(Vector_MRSkyPatch value) {return new(value);}
        }

        /// This is used for optional parameters of class `Vector_MRSkyPatch` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector_MRSkyPatch`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRSkyPatch`/`Const_Vector_MRSkyPatch` to pass it to the function.
        public class _InOptConst_Vector_MRSkyPatch
        {
            public Const_Vector_MRSkyPatch? Opt;

            public _InOptConst_Vector_MRSkyPatch() {}
            public _InOptConst_Vector_MRSkyPatch(Const_Vector_MRSkyPatch value) {Opt = value;}
            public static implicit operator _InOptConst_Vector_MRSkyPatch(Const_Vector_MRSkyPatch value) {return new(value);}
        }
    }
}

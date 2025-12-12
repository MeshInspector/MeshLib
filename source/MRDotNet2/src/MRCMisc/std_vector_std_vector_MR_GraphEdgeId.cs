public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::vector<std::vector<MR::GraphEdgeId>>`.
        /// This is the const half of the class.
        public class Const_Vector_StdVectorMRGraphEdgeId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Vector_StdVectorMRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Destroy(_Underlying *_this);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Vector_StdVectorMRGraphEdgeId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Vector_StdVectorMRGraphEdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Vector_StdVectorMRGraphEdgeId(MR.Std._ByValue_Vector_StdVectorMRGraphEdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Size", ExactSpelling = true)]
                extern static ulong __MR_std_vector_std_vector_MR_GraphEdgeId_Size(_Underlying *_this);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_vector_MR_GraphEdgeId_IsEmpty(_Underlying *_this);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// The memory capacity, measued in the number of elements.
            public unsafe ulong Capacity()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Capacity", ExactSpelling = true)]
                extern static ulong __MR_std_vector_std_vector_MR_GraphEdgeId_Capacity(_Underlying *_this);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_Capacity(_UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Std.Const_Vector_MRGraphEdgeId At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_At", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_At(_Underlying *_this, ulong i);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, read-only.
            public unsafe MR.Std.Const_Vector_MRGraphEdgeId? Front()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Front", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_Front(_Underlying *_this);
                var __ret = __MR_std_vector_std_vector_MR_GraphEdgeId_Front(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_Vector_MRGraphEdgeId(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, read-only.
            public unsafe MR.Std.Const_Vector_MRGraphEdgeId? Back()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Back", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_Back(_Underlying *_this);
                var __ret = __MR_std_vector_std_vector_MR_GraphEdgeId_Back(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_Vector_MRGraphEdgeId(__ret, is_owning: false) : null;
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Begin", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_Begin(_Underlying *_this);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_vector_MR_GraphEdgeId_IsBegin(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_End", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_End(_Underlying *_this);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_vector_MR_GraphEdgeId_IsEnd(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Convert a const iterator to an index.
            public unsafe long ToIndex(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_ToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_std_vector_MR_GraphEdgeId_ToIndex(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_ToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Convert a mutable iterator to an index.
            public unsafe long MutableToIndex(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_std_vector_MR_GraphEdgeId_MutableToIndex(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_MutableToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Read-only iterator for `MR_std_vector_std_vector_MR_GraphEdgeId`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother(MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable(MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator.
                public unsafe MR.Std.Const_Vector_MRGraphEdgeId Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two const iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator a, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Distance(MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *a, MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *b);
                    return __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Read-only iterator for `MR_std_vector_std_vector_MR_GraphEdgeId`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother(MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *other);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable(MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a const iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_const_iterator_OffsetBy(_UnderlyingPtr, delta);
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
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter) {return new MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_std_vector_std_vector_MR_GraphEdgeId`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother(MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator.
                public unsafe MR.Std.Vector_MRGraphEdgeId Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }

                /// Computes the signed difference between two mutable iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator a, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Distance(MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *a, MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *b);
                    return __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Mutable iterator for `MR_std_vector_std_vector_MR_GraphEdgeId`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother(MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *other);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a mutable iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a mutable iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_std_vector_MR_GraphEdgeId_iterator_OffsetBy(_UnderlyingPtr, delta);
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

        /// Generated from C++ container `std::vector<std::vector<MR::GraphEdgeId>>`.
        /// This is the non-const half of the class.
        public class Vector_StdVectorMRGraphEdgeId : Const_Vector_StdVectorMRGraphEdgeId
        {
            internal unsafe Vector_StdVectorMRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Vector_StdVectorMRGraphEdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Vector_StdVectorMRGraphEdgeId(MR.Std._ByValue_Vector_StdVectorMRGraphEdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_std_vector_MR_GraphEdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Vector_StdVectorMRGraphEdgeId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_StdVectorMRGraphEdgeId._Underlying *other);
                __MR_std_vector_std_vector_MR_GraphEdgeId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Resizes the container. The new elements if any are zeroed.
            public unsafe void Resize(ulong new_size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Resize", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Resize(_Underlying *_this, ulong new_size);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Resize(_UnderlyingPtr, new_size);
            }

            /// Resizes the container. The new elements if any are set to the specified value.
            public unsafe void ResizeWithDefaultValue(ulong new_size, MR.Std.Const_Vector_MRGraphEdgeId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_ResizeWithDefaultValue", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_ResizeWithDefaultValue(_Underlying *_this, ulong new_size, MR.Std.Const_Vector_MRGraphEdgeId._Underlying *value);
                __MR_std_vector_std_vector_MR_GraphEdgeId_ResizeWithDefaultValue(_UnderlyingPtr, new_size, value._UnderlyingPtr);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Clear", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Clear(_Underlying *_this);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Clear(_UnderlyingPtr);
            }

            /// Reserves memory for a certain number of elements. Never shrinks the memory.
            public unsafe void Reserve(ulong new_capacity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Reserve", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Reserve(_Underlying *_this, ulong new_capacity);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Reserve(_UnderlyingPtr, new_capacity);
            }

            /// Shrinks the capacity to match the size.
            public unsafe void ShrinkToFit()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_ShrinkToFit", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_ShrinkToFit(_Underlying *_this);
                __MR_std_vector_std_vector_MR_GraphEdgeId_ShrinkToFit(_UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.Std.Vector_MRGraphEdgeId MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableAt", ExactSpelling = true)]
                extern static MR.Std.Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// The first element or null if empty, mutable.
            public unsafe MR.Std.Vector_MRGraphEdgeId? MutableFront()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableFront", ExactSpelling = true)]
                extern static MR.Std.Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_MutableFront(_Underlying *_this);
                var __ret = __MR_std_vector_std_vector_MR_GraphEdgeId_MutableFront(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Vector_MRGraphEdgeId(__ret, is_owning: false) : null;
            }

            /// The last element or null if empty, mutable.
            public unsafe MR.Std.Vector_MRGraphEdgeId? MutableBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableBack", ExactSpelling = true)]
                extern static MR.Std.Vector_MRGraphEdgeId._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_MutableBack(_Underlying *_this);
                var __ret = __MR_std_vector_std_vector_MR_GraphEdgeId_MutableBack(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Vector_MRGraphEdgeId(__ret, is_owning: false) : null;
            }

            /// Inserts a new element at the end.
            public unsafe void PushBack(MR.Std._ByValue_Vector_MRGraphEdgeId new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_PushBack", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_PushBack(_Underlying *_this, MR.Misc._PassBy new_elem_pass_by, MR.Std.Vector_MRGraphEdgeId._Underlying *new_elem);
                __MR_std_vector_std_vector_MR_GraphEdgeId_PushBack(_UnderlyingPtr, new_elem.PassByMode, new_elem.Value is not null ? new_elem.Value._UnderlyingPtr : null);
            }

            /// Removes one element from the end.
            public unsafe void PopBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_PopBack", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_PopBack(_Underlying *_this);
                __MR_std_vector_std_vector_MR_GraphEdgeId_PopBack(_UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void Insert(ulong position, MR.Std._ByValue_Vector_MRGraphEdgeId new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Insert", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Insert(_Underlying *_this, ulong position, MR.Misc._PassBy new_elem_pass_by, MR.Std.Vector_MRGraphEdgeId._Underlying *new_elem);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Insert(_UnderlyingPtr, position, new_elem.PassByMode, new_elem.Value is not null ? new_elem.Value._UnderlyingPtr : null);
            }

            /// Erases the element at the specified position.
            public unsafe void Erase(ulong position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_Erase", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_Erase(_Underlying *_this, ulong position);
                __MR_std_vector_std_vector_MR_GraphEdgeId_Erase(_UnderlyingPtr, position);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void InsertAtMutableIter(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator position, MR.Std._ByValue_Vector_MRGraphEdgeId new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtMutableIter(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *position, MR.Misc._PassBy new_elem_pass_by, MR.Std.Vector_MRGraphEdgeId._Underlying *new_elem);
                __MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem.PassByMode, new_elem.Value is not null ? new_elem.Value._UnderlyingPtr : null);
            }

            /// Erases the element at the specified position.
            public unsafe void EraseAtMutableIter(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtMutableIter(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *position);
                __MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void InsertAtIter(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator position, MR.Std._ByValue_Vector_MRGraphEdgeId new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtIter(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *position, MR.Misc._PassBy new_elem_pass_by, MR.Std.Vector_MRGraphEdgeId._Underlying *new_elem);
                __MR_std_vector_std_vector_MR_GraphEdgeId_InsertAtIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem.PassByMode, new_elem.Value is not null ? new_elem.Value._UnderlyingPtr : null);
            }

            /// Erases the element at the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void EraseAtIter(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_ConstIterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtIter(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.ConstIterator._Underlying *position);
                __MR_std_vector_std_vector_MR_GraphEdgeId_EraseAtIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableBegin", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_MutableBegin(_Underlying *_this);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableBegin(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_MutableEnd", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMRGraphEdgeId.Iterator._Underlying *__MR_std_vector_std_vector_MR_GraphEdgeId_MutableEnd(_Underlying *_this);
                return new(__MR_std_vector_std_vector_MR_GraphEdgeId_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableEnd(_Underlying *_this, MR.Std.Vector_StdVectorMRGraphEdgeId.Const_Iterator._Underlying *iter);
                return __MR_std_vector_std_vector_MR_GraphEdgeId_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Vector_StdVectorMRGraphEdgeId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Vector_StdVectorMRGraphEdgeId`/`Const_Vector_StdVectorMRGraphEdgeId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Vector_StdVectorMRGraphEdgeId
        {
            internal readonly Const_Vector_StdVectorMRGraphEdgeId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Vector_StdVectorMRGraphEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Vector_StdVectorMRGraphEdgeId(Const_Vector_StdVectorMRGraphEdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Vector_StdVectorMRGraphEdgeId(Const_Vector_StdVectorMRGraphEdgeId arg) {return new(arg);}
            public _ByValue_Vector_StdVectorMRGraphEdgeId(MR.Misc._Moved<Vector_StdVectorMRGraphEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Vector_StdVectorMRGraphEdgeId(MR.Misc._Moved<Vector_StdVectorMRGraphEdgeId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Vector_StdVectorMRGraphEdgeId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector_StdVectorMRGraphEdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_StdVectorMRGraphEdgeId`/`Const_Vector_StdVectorMRGraphEdgeId` directly.
        public class _InOptMut_Vector_StdVectorMRGraphEdgeId
        {
            public Vector_StdVectorMRGraphEdgeId? Opt;

            public _InOptMut_Vector_StdVectorMRGraphEdgeId() {}
            public _InOptMut_Vector_StdVectorMRGraphEdgeId(Vector_StdVectorMRGraphEdgeId value) {Opt = value;}
            public static implicit operator _InOptMut_Vector_StdVectorMRGraphEdgeId(Vector_StdVectorMRGraphEdgeId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Vector_StdVectorMRGraphEdgeId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector_StdVectorMRGraphEdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_StdVectorMRGraphEdgeId`/`Const_Vector_StdVectorMRGraphEdgeId` to pass it to the function.
        public class _InOptConst_Vector_StdVectorMRGraphEdgeId
        {
            public Const_Vector_StdVectorMRGraphEdgeId? Opt;

            public _InOptConst_Vector_StdVectorMRGraphEdgeId() {}
            public _InOptConst_Vector_StdVectorMRGraphEdgeId(Const_Vector_StdVectorMRGraphEdgeId value) {Opt = value;}
            public static implicit operator _InOptConst_Vector_StdVectorMRGraphEdgeId(Const_Vector_StdVectorMRGraphEdgeId value) {return new(value);}
        }
    }
}

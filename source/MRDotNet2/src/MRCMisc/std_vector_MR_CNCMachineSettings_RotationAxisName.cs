public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::vector<MR::CNCMachineSettings::RotationAxisName>`.
        /// This is the const half of the class.
        public class Const_Vector_MRCNCMachineSettingsRotationAxisName : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Vector_MRCNCMachineSettingsRotationAxisName(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Destroy", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Destroy(_Underlying *_this);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Vector_MRCNCMachineSettingsRotationAxisName() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Vector_MRCNCMachineSettingsRotationAxisName() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Vector_MRCNCMachineSettingsRotationAxisName(MR.Std._ByValue_Vector_MRCNCMachineSettingsRotationAxisName other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Construct from a range of elements.
            public unsafe Const_Vector_MRCNCMachineSettingsRotationAxisName(MR.CNCMachineSettings.RotationAxisName? ptr, ulong size) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange(MR.CNCMachineSettings.RotationAxisName *ptr, ulong size);
                MR.CNCMachineSettings.RotationAxisName __deref_ptr = ptr.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange(ptr.HasValue ? &__deref_ptr : null, size);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Size", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Size(_Underlying *_this);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEmpty(_Underlying *_this);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// The memory capacity, measued in the number of elements.
            public unsafe ulong Capacity()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Capacity", ExactSpelling = true)]
                extern static ulong __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Capacity(_Underlying *_this);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Capacity(_UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.CNCMachineSettings.RotationAxisName At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_At", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_At(_Underlying *_this, ulong i);
                return *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_At(_UnderlyingPtr, i);
            }

            /// The first element or null if empty, read-only.
            public unsafe MR.CNCMachineSettings.RotationAxisName? Front()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Front", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Front(_Underlying *_this);
                var __ret = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Front(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// The last element or null if empty, read-only.
            public unsafe MR.CNCMachineSettings.RotationAxisName? Back()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Back", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Back(_Underlying *_this);
                var __ret = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Back(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            /// Returns a read-only pointer.
            public unsafe MR.CNCMachineSettings.RotationAxisName *Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Data", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Data(_Underlying *_this);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Data(_UnderlyingPtr);
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Begin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Begin(_Underlying *_this);
                return new(__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsBegin(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_End", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_End(_Underlying *_this);
                return new(__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEnd(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Convert a const iterator to an index.
            public unsafe long ToIndex(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ToIndex(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Convert a mutable iterator to an index.
            public unsafe long MutableToIndex(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableToIndex", ExactSpelling = true)]
                extern static long __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableToIndex(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableToIndex(_UnderlyingPtr, iter._UnderlyingPtr);
            }

            /// Read-only iterator for `MR_std_vector_MR_CNCMachineSettings_RotationAxisName`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator.
                public unsafe MR.CNCMachineSettings.RotationAxisName Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Deref(_Underlying *_this);
                    return *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Deref(_UnderlyingPtr);
                }

                /// Computes the signed difference between two const iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator a, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Distance(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *a, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *b);
                    return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Read-only iterator for `MR_std_vector_MR_CNCMachineSettings_RotationAxisName`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *other);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a const iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_const_iterator_OffsetBy(_UnderlyingPtr, delta);
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
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter) {return new MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_std_vector_MR_CNCMachineSettings_RotationAxisName`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Destroy(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator.
                public unsafe ref MR.CNCMachineSettings.RotationAxisName Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Deref", ExactSpelling = true)]
                    extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Deref(_Underlying *_this);
                    return ref *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Deref(_UnderlyingPtr);
                }

                /// Computes the signed difference between two mutable iterators. Completes in constant time.
                public static unsafe long Distance(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator a, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Distance", ExactSpelling = true)]
                    extern static long __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Distance(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *a, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *b);
                    return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Distance(a._UnderlyingPtr, b._UnderlyingPtr);
                }
            }

            /// Mutable iterator for `MR_std_vector_MR_CNCMachineSettings_RotationAxisName`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *other);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Incr(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a mutable iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Decr(_Underlying *_this);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_Decr(_UnderlyingPtr);
                }

                /// Increments or decrements a mutable iterator by the specific amount. Completes in constant time.
                public unsafe void OffsetBy(long delta)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_OffsetBy", ExactSpelling = true)]
                    extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_OffsetBy(_Underlying *_this, long delta);
                    __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_iterator_OffsetBy(_UnderlyingPtr, delta);
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

        /// Generated from C++ container `std::vector<MR::CNCMachineSettings::RotationAxisName>`.
        /// This is the non-const half of the class.
        public class Vector_MRCNCMachineSettingsRotationAxisName : Const_Vector_MRCNCMachineSettingsRotationAxisName
        {
            internal unsafe Vector_MRCNCMachineSettingsRotationAxisName(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Vector_MRCNCMachineSettingsRotationAxisName() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct();
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Vector_MRCNCMachineSettingsRotationAxisName(MR.Std._ByValue_Vector_MRCNCMachineSettingsRotationAxisName other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *other);
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Vector_MRCNCMachineSettingsRotationAxisName other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *other);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Construct from a range of elements.
            public unsafe Vector_MRCNCMachineSettingsRotationAxisName(MR.CNCMachineSettings.RotationAxisName? ptr, ulong size) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange(MR.CNCMachineSettings.RotationAxisName *ptr, ulong size);
                MR.CNCMachineSettings.RotationAxisName __deref_ptr = ptr.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ConstructFromRange(ptr.HasValue ? &__deref_ptr : null, size);
            }

            /// Assign from a range of elements, overwriting previous contents.
            public unsafe void Assign(MR.CNCMachineSettings.RotationAxisName? ptr, ulong size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromRange", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromRange(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName *ptr, ulong size);
                MR.CNCMachineSettings.RotationAxisName __deref_ptr = ptr.GetValueOrDefault();
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_AssignFromRange(_UnderlyingPtr, ptr.HasValue ? &__deref_ptr : null, size);
            }

            /// Resizes the container. The new elements if any are zeroed.
            public unsafe void Resize(ulong new_size)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Resize", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Resize(_Underlying *_this, ulong new_size);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Resize(_UnderlyingPtr, new_size);
            }

            /// Resizes the container. The new elements if any are set to the specified value.
            public unsafe void ResizeWithDefaultValue(ulong new_size, MR.CNCMachineSettings.RotationAxisName value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ResizeWithDefaultValue", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ResizeWithDefaultValue(_Underlying *_this, ulong new_size, MR.CNCMachineSettings.RotationAxisName value);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ResizeWithDefaultValue(_UnderlyingPtr, new_size, value);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Clear", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Clear(_Underlying *_this);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Clear(_UnderlyingPtr);
            }

            /// Reserves memory for a certain number of elements. Never shrinks the memory.
            public unsafe void Reserve(ulong new_capacity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Reserve", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Reserve(_Underlying *_this, ulong new_capacity);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Reserve(_UnderlyingPtr, new_capacity);
            }

            /// Shrinks the capacity to match the size.
            public unsafe void ShrinkToFit()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ShrinkToFit", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ShrinkToFit(_Underlying *_this);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_ShrinkToFit(_UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe ref MR.CNCMachineSettings.RotationAxisName MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableAt", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableAt(_Underlying *_this, ulong i);
                return ref *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableAt(_UnderlyingPtr, i);
            }

            /// The first element or null if empty, mutable.
            public unsafe MR.Misc.Ref<MR.CNCMachineSettings.RotationAxisName>? MutableFront()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableFront", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableFront(_Underlying *_this);
                var __ret = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableFront(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<MR.CNCMachineSettings.RotationAxisName>(__ret) : null;
            }

            /// The last element or null if empty, mutable.
            public unsafe MR.Misc.Ref<MR.CNCMachineSettings.RotationAxisName>? MutableBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBack", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBack(_Underlying *_this);
                var __ret = __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBack(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<MR.CNCMachineSettings.RotationAxisName>(__ret) : null;
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            /// Returns a mutable pointer.
            public unsafe MR.CNCMachineSettings.RotationAxisName *MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableData", ExactSpelling = true)]
                extern static MR.CNCMachineSettings.RotationAxisName *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableData(_Underlying *_this);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableData(_UnderlyingPtr);
            }

            /// Inserts a new element at the end.
            public unsafe void PushBack(MR.CNCMachineSettings.RotationAxisName new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PushBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PushBack(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName new_elem);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PushBack(_UnderlyingPtr, new_elem);
            }

            /// Removes one element from the end.
            public unsafe void PopBack()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PopBack", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PopBack(_Underlying *_this);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_PopBack(_UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void Insert(ulong position, MR.CNCMachineSettings.RotationAxisName new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Insert", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Insert(_Underlying *_this, ulong position, MR.CNCMachineSettings.RotationAxisName new_elem);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Insert(_UnderlyingPtr, position, new_elem);
            }

            /// Erases the element at the specified position.
            public unsafe void Erase(ulong position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Erase", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Erase(_Underlying *_this, ulong position);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_Erase(_UnderlyingPtr, position);
            }

            /// Inserts a new element right before the specified position.
            public unsafe void InsertAtMutableIter(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator position, MR.CNCMachineSettings.RotationAxisName new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtMutableIter(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *position, MR.CNCMachineSettings.RotationAxisName new_elem);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem);
            }

            /// Erases the element at the specified position.
            public unsafe void EraseAtMutableIter(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtMutableIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtMutableIter(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *position);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtMutableIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// Inserts a new element right before the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void InsertAtIter(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator position, MR.CNCMachineSettings.RotationAxisName new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtIter(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *position, MR.CNCMachineSettings.RotationAxisName new_elem);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_InsertAtIter(_UnderlyingPtr, position._UnderlyingPtr, new_elem);
            }

            /// Erases the element at the specified position. This version takes the position in form of a const iterator, that's the only difference.
            public unsafe void EraseAtIter(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_ConstIterator position)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtIter", ExactSpelling = true)]
                extern static void __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtIter(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.ConstIterator._Underlying *position);
                __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_EraseAtIter(_UnderlyingPtr, position._UnderlyingPtr);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBegin", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBegin(_Underlying *_this);
                return new(__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableBegin(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableEnd", ExactSpelling = true)]
                extern static MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Iterator._Underlying *__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableEnd(_Underlying *_this);
                return new(__MR_std_vector_MR_CNCMachineSettings_RotationAxisName_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableEnd(_Underlying *_this, MR.Std.Vector_MRCNCMachineSettingsRotationAxisName.Const_Iterator._Underlying *iter);
                return __MR_std_vector_MR_CNCMachineSettings_RotationAxisName_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Vector_MRCNCMachineSettingsRotationAxisName` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Vector_MRCNCMachineSettingsRotationAxisName`/`Const_Vector_MRCNCMachineSettingsRotationAxisName` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Vector_MRCNCMachineSettingsRotationAxisName
        {
            internal readonly Const_Vector_MRCNCMachineSettingsRotationAxisName? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Vector_MRCNCMachineSettingsRotationAxisName() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Vector_MRCNCMachineSettingsRotationAxisName(Const_Vector_MRCNCMachineSettingsRotationAxisName new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Vector_MRCNCMachineSettingsRotationAxisName(Const_Vector_MRCNCMachineSettingsRotationAxisName arg) {return new(arg);}
            public _ByValue_Vector_MRCNCMachineSettingsRotationAxisName(MR.Misc._Moved<Vector_MRCNCMachineSettingsRotationAxisName> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Vector_MRCNCMachineSettingsRotationAxisName(MR.Misc._Moved<Vector_MRCNCMachineSettingsRotationAxisName> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Vector_MRCNCMachineSettingsRotationAxisName` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector_MRCNCMachineSettingsRotationAxisName`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRCNCMachineSettingsRotationAxisName`/`Const_Vector_MRCNCMachineSettingsRotationAxisName` directly.
        public class _InOptMut_Vector_MRCNCMachineSettingsRotationAxisName
        {
            public Vector_MRCNCMachineSettingsRotationAxisName? Opt;

            public _InOptMut_Vector_MRCNCMachineSettingsRotationAxisName() {}
            public _InOptMut_Vector_MRCNCMachineSettingsRotationAxisName(Vector_MRCNCMachineSettingsRotationAxisName value) {Opt = value;}
            public static implicit operator _InOptMut_Vector_MRCNCMachineSettingsRotationAxisName(Vector_MRCNCMachineSettingsRotationAxisName value) {return new(value);}
        }

        /// This is used for optional parameters of class `Vector_MRCNCMachineSettingsRotationAxisName` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector_MRCNCMachineSettingsRotationAxisName`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Vector_MRCNCMachineSettingsRotationAxisName`/`Const_Vector_MRCNCMachineSettingsRotationAxisName` to pass it to the function.
        public class _InOptConst_Vector_MRCNCMachineSettingsRotationAxisName
        {
            public Const_Vector_MRCNCMachineSettingsRotationAxisName? Opt;

            public _InOptConst_Vector_MRCNCMachineSettingsRotationAxisName() {}
            public _InOptConst_Vector_MRCNCMachineSettingsRotationAxisName(Const_Vector_MRCNCMachineSettingsRotationAxisName value) {Opt = value;}
            public static implicit operator _InOptConst_Vector_MRCNCMachineSettingsRotationAxisName(Const_Vector_MRCNCMachineSettingsRotationAxisName value) {return new(value);}
        }
    }
}

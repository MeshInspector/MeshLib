public static partial class MR
{
    public static partial class Std
    {
        /// Generated from C++ container `std::set<std::string>`.
        /// This is the const half of the class.
        public class Const_Set_StdString : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Set_StdString(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Destroy", ExactSpelling = true)]
                extern static void __MR_std_set_std_string_Destroy(_Underlying *_this);
                __MR_std_set_std_string_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Set_StdString() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Set_StdString() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Set_StdString._Underlying *__MR_std_set_std_string_DefaultConstruct();
                _UnderlyingPtr = __MR_std_set_std_string_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Set_StdString(MR.Std._ByValue_Set_StdString other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Set_StdString._Underlying *__MR_std_set_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Set_StdString._Underlying *other);
                _UnderlyingPtr = __MR_std_set_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Size", ExactSpelling = true)]
                extern static ulong __MR_std_set_std_string_Size(_Underlying *_this);
                return __MR_std_set_std_string_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_std_set_std_string_IsEmpty(_Underlying *_this);
                return __MR_std_set_std_string_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// Checks if the contain contains this key.
            public unsafe bool Contains(ReadOnlySpan<char> key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Contains", ExactSpelling = true)]
                extern static byte __MR_std_set_std_string_Contains(_Underlying *_this, byte *key, byte *key_end);
                byte[] __bytes_key = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(key.Length)];
                int __len_key = System.Text.Encoding.UTF8.GetBytes(key, __bytes_key);
                fixed (byte *__ptr_key = __bytes_key)
                {
                    return __MR_std_set_std_string_Contains(_UnderlyingPtr, __ptr_key, __ptr_key + __len_key) != 0;
                }
            }

            /// Finds the element by key, or returns the end iterator if no such key. Returns a read-only iterator.
            public unsafe MR.Std.Set_StdString.ConstIterator Find(ReadOnlySpan<char> key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Find", ExactSpelling = true)]
                extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_Find(_Underlying *_this, byte *key, byte *key_end);
                byte[] __bytes_key = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(key.Length)];
                int __len_key = System.Text.Encoding.UTF8.GetBytes(key, __bytes_key);
                fixed (byte *__ptr_key = __bytes_key)
                {
                    return new(__MR_std_set_std_string_Find(_UnderlyingPtr, __ptr_key, __ptr_key + __len_key), is_owning: true);
                }
            }

            /// The begin iterator, const.
            public unsafe MR.Std.Set_StdString.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Begin", ExactSpelling = true)]
                extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_Begin(_Underlying *_this);
                return new(__MR_std_set_std_string_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Std.Set_StdString.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_IsBegin", ExactSpelling = true)]
                extern static byte __MR_std_set_std_string_IsBegin(_Underlying *_this, MR.Std.Set_StdString.Const_ConstIterator._Underlying *iter);
                return __MR_std_set_std_string_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Std.Set_StdString.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_End", ExactSpelling = true)]
                extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_End(_Underlying *_this);
                return new(__MR_std_set_std_string_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Std.Set_StdString.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_IsEnd", ExactSpelling = true)]
                extern static byte __MR_std_set_std_string_IsEnd(_Underlying *_this, MR.Std.Set_StdString.Const_ConstIterator._Underlying *iter);
                return __MR_std_set_std_string_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Read-only iterator for `MR_std_set_std_string`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_set_std_string_const_iterator_Destroy(_Underlying *_this);
                    __MR_std_set_std_string_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_set_std_string_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Std.Set_StdString.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_const_iterator_ConstructFromAnother(MR.Std.Set_StdString.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_set_std_string_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a const iterator.
                public unsafe MR.Std.Const_String Deref()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_Deref", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_std_set_std_string_const_iterator_Deref(_Underlying *_this);
                    return new(__MR_std_set_std_string_const_iterator_Deref(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Read-only iterator for `MR_std_set_std_string`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_set_std_string_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Std.Set_StdString.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Set_StdString.ConstIterator._Underlying *__MR_std_set_std_string_const_iterator_ConstructFromAnother(MR.Std.Set_StdString.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_std_set_std_string_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Set_StdString.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_set_std_string_const_iterator_AssignFromAnother(_Underlying *_this, MR.Std.Set_StdString.ConstIterator._Underlying *other);
                    __MR_std_set_std_string_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_std_set_std_string_const_iterator_Incr(_Underlying *_this);
                    __MR_std_set_std_string_const_iterator_Incr(_UnderlyingPtr);
                }

                /// Decrements a const iterator.
                public unsafe void Decr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_const_iterator_Decr", ExactSpelling = true)]
                    extern static void __MR_std_set_std_string_const_iterator_Decr(_Underlying *_this);
                    __MR_std_set_std_string_const_iterator_Decr(_UnderlyingPtr);
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
            }
        }

        /// Generated from C++ container `std::set<std::string>`.
        /// This is the non-const half of the class.
        public class Set_StdString : Const_Set_StdString
        {
            internal unsafe Set_StdString(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Set_StdString() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Set_StdString._Underlying *__MR_std_set_std_string_DefaultConstruct();
                _UnderlyingPtr = __MR_std_set_std_string_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Set_StdString(MR.Std._ByValue_Set_StdString other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Set_StdString._Underlying *__MR_std_set_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Set_StdString._Underlying *other);
                _UnderlyingPtr = __MR_std_set_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Set_StdString other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_set_std_string_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Set_StdString._Underlying *other);
                __MR_std_set_std_string_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Clear", ExactSpelling = true)]
                extern static void __MR_std_set_std_string_Clear(_Underlying *_this);
                __MR_std_set_std_string_Clear(_UnderlyingPtr);
            }

            /// Inserts a new element.
            public unsafe void Insert(ReadOnlySpan<char> new_elem)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_set_std_string_Insert", ExactSpelling = true)]
                extern static void __MR_std_set_std_string_Insert(_Underlying *_this, byte *new_elem, byte *new_elem_end);
                byte[] __bytes_new_elem = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(new_elem.Length)];
                int __len_new_elem = System.Text.Encoding.UTF8.GetBytes(new_elem, __bytes_new_elem);
                fixed (byte *__ptr_new_elem = __bytes_new_elem)
                {
                    __MR_std_set_std_string_Insert(_UnderlyingPtr, __ptr_new_elem, __ptr_new_elem + __len_new_elem);
                }
            }
        }

        /// This is used as a function parameter when the underlying function receives `Set_StdString` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Set_StdString`/`Const_Set_StdString` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Set_StdString
        {
            internal readonly Const_Set_StdString? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Set_StdString() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Set_StdString(Const_Set_StdString new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Set_StdString(Const_Set_StdString arg) {return new(arg);}
            public _ByValue_Set_StdString(MR.Misc._Moved<Set_StdString> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Set_StdString(MR.Misc._Moved<Set_StdString> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Set_StdString` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Set_StdString`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Set_StdString`/`Const_Set_StdString` directly.
        public class _InOptMut_Set_StdString
        {
            public Set_StdString? Opt;

            public _InOptMut_Set_StdString() {}
            public _InOptMut_Set_StdString(Set_StdString value) {Opt = value;}
            public static implicit operator _InOptMut_Set_StdString(Set_StdString value) {return new(value);}
        }

        /// This is used for optional parameters of class `Set_StdString` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Set_StdString`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Set_StdString`/`Const_Set_StdString` to pass it to the function.
        public class _InOptConst_Set_StdString
        {
            public Const_Set_StdString? Opt;

            public _InOptConst_Set_StdString() {}
            public _InOptConst_Set_StdString(Const_Set_StdString value) {Opt = value;}
            public static implicit operator _InOptConst_Set_StdString(Const_Set_StdString value) {return new(value);}
        }
    }
}

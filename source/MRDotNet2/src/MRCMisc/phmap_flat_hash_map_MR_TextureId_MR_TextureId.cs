public static partial class MR
{
    public static partial class Phmap
    {
        /// Generated from C++ container `phmap::flat_hash_map<MR::TextureId, MR::TextureId>`.
        /// This is the const half of the class.
        public class Const_FlatHashMap_MRTextureId_MRTextureId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_FlatHashMap_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Destroy", ExactSpelling = true)]
                extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Destroy(_Underlying *_this);
                __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_FlatHashMap_MRTextureId_MRTextureId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_FlatHashMap_MRTextureId_MRTextureId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct();
                _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_FlatHashMap_MRTextureId_MRTextureId(MR.Phmap._ByValue_FlatHashMap_MRTextureId_MRTextureId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *other);
                _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The number of elements.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Size", ExactSpelling = true)]
                extern static ulong __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Size(_Underlying *_this);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Size(_UnderlyingPtr);
            }

            /// Returns true if the size is zero.
            public unsafe bool IsEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEmpty", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEmpty(_Underlying *_this);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEmpty(_UnderlyingPtr) != 0;
            }

            /// Checks if the contain contains this key.
            public unsafe bool Contains(MR.Const_TextureId key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Contains", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Contains(_Underlying *_this, MR.Const_TextureId._Underlying *key);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Contains(_UnderlyingPtr, key._UnderlyingPtr) != 0;
            }

            /// Finds the element by key, or returns the end iterator if no such key. Returns a read-only iterator.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator Find(MR.Const_TextureId key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Find", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Find(_Underlying *_this, MR.Const_TextureId._Underlying *key);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Find(_UnderlyingPtr, key._UnderlyingPtr), is_owning: true);
            }

            /// The begin iterator, const.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator Begin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Begin", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Begin(_Underlying *_this);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Begin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the begin iterator.
            public unsafe bool IsBegin(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsBegin", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsBegin(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator._Underlying *iter);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, const.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator End()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_End", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_End(_Underlying *_this);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_End(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a const iterator is the end iterator.
            public unsafe bool IsEnd(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEnd", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEnd(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator._Underlying *iter);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// Read-only iterator for `MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Destroy(_Underlying *_this);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConstIterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe Const_ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator Const_ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter) {return new(iter);}

                /// Dereferences a const iterator, returning the key.
                public unsafe MR.Const_TextureId DerefKey()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefKey", ExactSpelling = true)]
                    extern static MR.Const_TextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefKey(_Underlying *_this);
                    return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefKey(_UnderlyingPtr), is_owning: false);
                }

                /// Dereferences a const iterator, returning the mapped value.
                public unsafe MR.Const_TextureId DerefValue()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefValue", ExactSpelling = true)]
                    extern static MR.Const_TextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefValue(_Underlying *_this);
                    return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DerefValue(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Read-only iterator for `MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId`.
            /// This is the non-const half of the class.
            public class ConstIterator : Const_ConstIterator
            {
                internal unsafe ConstIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConstIterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *other);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_ConstIterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_AssignFromAnother(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *other);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public unsafe ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *iter);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_FromMutable(iter._UnderlyingPtr);
                }

                /// Makes a const iterator from a mutable one.
                public static unsafe implicit operator ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter) {return new(iter);}

                /// Increments a const iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Incr(_Underlying *_this);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_const_iterator_Incr(_UnderlyingPtr);
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
                public static unsafe implicit operator _InOptConst_ConstIterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter) {return new MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.ConstIterator(iter);}
            }

            /// Mutable iterator for `MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId`.
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
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Destroy", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Destroy(_Underlying *_this);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Iterator() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Iterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Dereferences a mutable iterator, returning the key.
                public unsafe MR.Const_TextureId DerefKey()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefKey", ExactSpelling = true)]
                    extern static MR.Const_TextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefKey(_Underlying *_this);
                    return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefKey(_UnderlyingPtr), is_owning: false);
                }

                /// Dereferences a mutable iterator, returning the mapped value.
                public unsafe MR.Mut_TextureId DerefValue()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefValue", ExactSpelling = true)]
                    extern static MR.Mut_TextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefValue(_Underlying *_this);
                    return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DerefValue(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Mutable iterator for `MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId`.
            /// This is the non-const half of the class.
            public class Iterator : Const_Iterator
            {
                internal unsafe Iterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Iterator() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct();
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Iterator(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *other);
                    _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_ConstructFromAnother(other._UnderlyingPtr);
                }

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_AssignFromAnother(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *other);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
                }

                /// Increments a mutable iterator.
                public unsafe void Incr()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Incr", ExactSpelling = true)]
                    extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Incr(_Underlying *_this);
                    __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_iterator_Incr(_UnderlyingPtr);
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

        /// Generated from C++ container `phmap::flat_hash_map<MR::TextureId, MR::TextureId>`.
        /// This is the non-const half of the class.
        public class FlatHashMap_MRTextureId_MRTextureId : Const_FlatHashMap_MRTextureId_MRTextureId
        {
            internal unsafe FlatHashMap_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe FlatHashMap_MRTextureId_MRTextureId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct();
                _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe FlatHashMap_MRTextureId_MRTextureId(MR.Phmap._ByValue_FlatHashMap_MRTextureId_MRTextureId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *other);
                _UnderlyingPtr = __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Phmap._ByValue_FlatHashMap_MRTextureId_MRTextureId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId._Underlying *other);
                __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Removes all elements from the container.
            public unsafe void Clear()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Clear", ExactSpelling = true)]
                extern static void __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Clear(_Underlying *_this);
                __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_Clear(_UnderlyingPtr);
            }

            /// Returns the element with the specific key. If it doesn't exist, creates it first. Acts like map's `operator[]` in C++.
            public unsafe MR.Mut_TextureId FindOrConstructElem(MR.Const_TextureId key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindOrConstructElem", ExactSpelling = true)]
                extern static MR.Mut_TextureId._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindOrConstructElem(_Underlying *_this, MR.Const_TextureId._Underlying *key);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindOrConstructElem(_UnderlyingPtr, key._UnderlyingPtr), is_owning: false);
            }

            /// Finds the element by key, or returns the end iterator if no such key. Returns a mutable iterator.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator FindMutable(MR.Const_TextureId key)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindMutable", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindMutable(_Underlying *_this, MR.Const_TextureId._Underlying *key);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_FindMutable(_UnderlyingPtr, key._UnderlyingPtr), is_owning: true);
            }

            /// The begin iterator, mutable.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator MutableBegin()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableBegin", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableBegin(_Underlying *_this);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableBegin(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the begin iterator.
            public unsafe bool IsMutableBegin(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableBegin", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableBegin(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator._Underlying *iter);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableBegin(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }

            /// The end iterator, mutable.
            public unsafe MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator MutableEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableEnd", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Iterator._Underlying *__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableEnd(_Underlying *_this);
                return new(__MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_MutableEnd(_UnderlyingPtr), is_owning: true);
            }

            /// Tests whether a mutable iterator is the end iterator.
            public unsafe bool IsMutableEnd(MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator iter)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableEnd", ExactSpelling = true)]
                extern static byte __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableEnd(_Underlying *_this, MR.Phmap.FlatHashMap_MRTextureId_MRTextureId.Const_Iterator._Underlying *iter);
                return __MR_phmap_flat_hash_map_MR_TextureId_MR_TextureId_IsMutableEnd(_UnderlyingPtr, iter._UnderlyingPtr) != 0;
            }
        }

        /// This is used as a function parameter when the underlying function receives `FlatHashMap_MRTextureId_MRTextureId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `FlatHashMap_MRTextureId_MRTextureId`/`Const_FlatHashMap_MRTextureId_MRTextureId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_FlatHashMap_MRTextureId_MRTextureId
        {
            internal readonly Const_FlatHashMap_MRTextureId_MRTextureId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_FlatHashMap_MRTextureId_MRTextureId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_FlatHashMap_MRTextureId_MRTextureId(Const_FlatHashMap_MRTextureId_MRTextureId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_FlatHashMap_MRTextureId_MRTextureId(Const_FlatHashMap_MRTextureId_MRTextureId arg) {return new(arg);}
            public _ByValue_FlatHashMap_MRTextureId_MRTextureId(MR.Misc._Moved<FlatHashMap_MRTextureId_MRTextureId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_FlatHashMap_MRTextureId_MRTextureId(MR.Misc._Moved<FlatHashMap_MRTextureId_MRTextureId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `FlatHashMap_MRTextureId_MRTextureId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_FlatHashMap_MRTextureId_MRTextureId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FlatHashMap_MRTextureId_MRTextureId`/`Const_FlatHashMap_MRTextureId_MRTextureId` directly.
        public class _InOptMut_FlatHashMap_MRTextureId_MRTextureId
        {
            public FlatHashMap_MRTextureId_MRTextureId? Opt;

            public _InOptMut_FlatHashMap_MRTextureId_MRTextureId() {}
            public _InOptMut_FlatHashMap_MRTextureId_MRTextureId(FlatHashMap_MRTextureId_MRTextureId value) {Opt = value;}
            public static implicit operator _InOptMut_FlatHashMap_MRTextureId_MRTextureId(FlatHashMap_MRTextureId_MRTextureId value) {return new(value);}
        }

        /// This is used for optional parameters of class `FlatHashMap_MRTextureId_MRTextureId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_FlatHashMap_MRTextureId_MRTextureId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FlatHashMap_MRTextureId_MRTextureId`/`Const_FlatHashMap_MRTextureId_MRTextureId` to pass it to the function.
        public class _InOptConst_FlatHashMap_MRTextureId_MRTextureId
        {
            public Const_FlatHashMap_MRTextureId_MRTextureId? Opt;

            public _InOptConst_FlatHashMap_MRTextureId_MRTextureId() {}
            public _InOptConst_FlatHashMap_MRTextureId_MRTextureId(Const_FlatHashMap_MRTextureId_MRTextureId value) {Opt = value;}
            public static implicit operator _InOptConst_FlatHashMap_MRTextureId_MRTextureId(Const_FlatHashMap_MRTextureId_MRTextureId value) {return new(value);}
        }
    }
}

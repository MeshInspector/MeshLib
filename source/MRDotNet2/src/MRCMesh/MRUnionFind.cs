public static partial class MR
{
    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::FaceId>`.
    /// This is the const half of the class.
    public class Const_UnionFind_MRFaceId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UnionFind_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_FaceId_Destroy(_Underlying *_this);
            __MR_UnionFind_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UnionFind_MRFaceId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UnionFind_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::FaceId>::UnionFind`.
        public unsafe Const_UnionFind_MRFaceId(MR._ByValue_UnionFind_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::FaceId>::UnionFind`.
        public unsafe Const_UnionFind_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_Construct(size);
        }

        /// returns the number of elements in union-find
        /// Generated from method `MR::UnionFind<MR::FaceId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_size", ExactSpelling = true)]
            extern static ulong __MR_UnionFind_MR_FaceId_size(_Underlying *_this);
            return __MR_UnionFind_MR_FaceId_size(_UnderlyingPtr);
        }

        /// returns true if given element is the root of some set
        /// Generated from method `MR::UnionFind<MR::FaceId>::isRoot`.
        public unsafe bool IsRoot(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_isRoot", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_FaceId_isRoot(_Underlying *_this, MR.FaceId a);
            return __MR_UnionFind_MR_FaceId_isRoot(_UnderlyingPtr, a) != 0;
        }

        /// return parent element of this element, which is equal to given element only for set's root
        /// Generated from method `MR::UnionFind<MR::FaceId>::parent`.
        public unsafe MR.FaceId Parent(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_parent", ExactSpelling = true)]
            extern static MR.FaceId __MR_UnionFind_MR_FaceId_parent(_Underlying *_this, MR.FaceId a);
            return __MR_UnionFind_MR_FaceId_parent(_UnderlyingPtr, a);
        }

        /// gets the parents of all elements as is
        /// Generated from method `MR::UnionFind<MR::FaceId>::parents`.
        public unsafe MR.Const_FaceMap Parents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_parents", ExactSpelling = true)]
            extern static MR.Const_FaceMap._Underlying *__MR_UnionFind_MR_FaceId_parents(_Underlying *_this);
            return new(__MR_UnionFind_MR_FaceId_parents(_UnderlyingPtr), is_owning: false);
        }
    }

    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::FaceId>`.
    /// This is the non-const half of the class.
    public class UnionFind_MRFaceId : Const_UnionFind_MRFaceId
    {
        internal unsafe UnionFind_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UnionFind_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::FaceId>::UnionFind`.
        public unsafe UnionFind_MRFaceId(MR._ByValue_UnionFind_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::FaceId>::UnionFind`.
        public unsafe UnionFind_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_FaceId_Construct(size);
        }

        /// Generated from method `MR::UnionFind<MR::FaceId>::operator=`.
        public unsafe MR.UnionFind_MRFaceId Assign(MR._ByValue_UnionFind_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_UnionFind_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRFaceId._Underlying *_other);
            return new(__MR_UnionFind_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// resets union-find to represent given number of elements, each element is the only one in its disjoint set
        /// Generated from method `MR::UnionFind<MR::FaceId>::reset`.
        public unsafe void Reset(ulong size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_reset", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_FaceId_reset(_Underlying *_this, ulong size);
            __MR_UnionFind_MR_FaceId_reset(_UnderlyingPtr, size);
        }

        /// unite two elements,
        /// \return first: new common root, second: true = union was done, false = first and second were already united
        /// Generated from method `MR::UnionFind<MR::FaceId>::unite`.
        public unsafe MR.Std.Pair_MRFaceId_Bool Unite(MR.FaceId first, MR.FaceId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_unite", ExactSpelling = true)]
            extern static MR.Std.Pair_MRFaceId_Bool._Underlying *__MR_UnionFind_MR_FaceId_unite(_Underlying *_this, MR.FaceId first, MR.FaceId second);
            return new(__MR_UnionFind_MR_FaceId_unite(_UnderlyingPtr, first, second), is_owning: true);
        }

        /// returns true if given two elements are from one set
        /// Generated from method `MR::UnionFind<MR::FaceId>::united`.
        public unsafe bool United(MR.FaceId first, MR.FaceId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_united", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_FaceId_united(_Underlying *_this, MR.FaceId first, MR.FaceId second);
            return __MR_UnionFind_MR_FaceId_united(_UnderlyingPtr, first, second) != 0;
        }

        /// finds the root of the set containing given element with optimizing data structure updates
        /// Generated from method `MR::UnionFind<MR::FaceId>::find`.
        public unsafe MR.FaceId Find(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_find", ExactSpelling = true)]
            extern static MR.FaceId __MR_UnionFind_MR_FaceId_find(_Underlying *_this, MR.FaceId a);
            return __MR_UnionFind_MR_FaceId_find(_UnderlyingPtr, a);
        }

        /// finds the root of the set containing given element with optimizing data structure in the range [begin, end)
        /// Generated from method `MR::UnionFind<MR::FaceId>::findUpdateRange`.
        public unsafe MR.FaceId FindUpdateRange(MR.FaceId a, MR.FaceId begin, MR.FaceId end)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_findUpdateRange", ExactSpelling = true)]
            extern static MR.FaceId __MR_UnionFind_MR_FaceId_findUpdateRange(_Underlying *_this, MR.FaceId a, MR.FaceId begin, MR.FaceId end);
            return __MR_UnionFind_MR_FaceId_findUpdateRange(_UnderlyingPtr, a, begin, end);
        }

        /// sets the root of corresponding set as the parent of each element, then returns the vector
        /// Generated from method `MR::UnionFind<MR::FaceId>::roots`.
        public unsafe MR.Const_FaceMap Roots()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_roots", ExactSpelling = true)]
            extern static MR.Const_FaceMap._Underlying *__MR_UnionFind_MR_FaceId_roots(_Underlying *_this);
            return new(__MR_UnionFind_MR_FaceId_roots(_UnderlyingPtr), is_owning: false);
        }

        /// returns the number of elements in the set containing given element
        /// Generated from method `MR::UnionFind<MR::FaceId>::sizeOfComp`.
        public unsafe int SizeOfComp(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_FaceId_sizeOfComp", ExactSpelling = true)]
            extern static int __MR_UnionFind_MR_FaceId_sizeOfComp(_Underlying *_this, MR.FaceId a);
            return __MR_UnionFind_MR_FaceId_sizeOfComp(_UnderlyingPtr, a);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UnionFind_MRFaceId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UnionFind_MRFaceId`/`Const_UnionFind_MRFaceId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UnionFind_MRFaceId
    {
        internal readonly Const_UnionFind_MRFaceId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UnionFind_MRFaceId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UnionFind_MRFaceId(Const_UnionFind_MRFaceId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UnionFind_MRFaceId(Const_UnionFind_MRFaceId arg) {return new(arg);}
        public _ByValue_UnionFind_MRFaceId(MR.Misc._Moved<UnionFind_MRFaceId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UnionFind_MRFaceId(MR.Misc._Moved<UnionFind_MRFaceId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UnionFind_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UnionFind_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRFaceId`/`Const_UnionFind_MRFaceId` directly.
    public class _InOptMut_UnionFind_MRFaceId
    {
        public UnionFind_MRFaceId? Opt;

        public _InOptMut_UnionFind_MRFaceId() {}
        public _InOptMut_UnionFind_MRFaceId(UnionFind_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_UnionFind_MRFaceId(UnionFind_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `UnionFind_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UnionFind_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRFaceId`/`Const_UnionFind_MRFaceId` to pass it to the function.
    public class _InOptConst_UnionFind_MRFaceId
    {
        public Const_UnionFind_MRFaceId? Opt;

        public _InOptConst_UnionFind_MRFaceId() {}
        public _InOptConst_UnionFind_MRFaceId(Const_UnionFind_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_UnionFind_MRFaceId(Const_UnionFind_MRFaceId value) {return new(value);}
    }

    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::VertId>`.
    /// This is the const half of the class.
    public class Const_UnionFind_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UnionFind_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_VertId_Destroy(_Underlying *_this);
            __MR_UnionFind_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UnionFind_MRVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UnionFind_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::VertId>::UnionFind`.
        public unsafe Const_UnionFind_MRVertId(MR._ByValue_UnionFind_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::VertId>::UnionFind`.
        public unsafe Const_UnionFind_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_Construct(size);
        }

        /// returns the number of elements in union-find
        /// Generated from method `MR::UnionFind<MR::VertId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_size", ExactSpelling = true)]
            extern static ulong __MR_UnionFind_MR_VertId_size(_Underlying *_this);
            return __MR_UnionFind_MR_VertId_size(_UnderlyingPtr);
        }

        /// returns true if given element is the root of some set
        /// Generated from method `MR::UnionFind<MR::VertId>::isRoot`.
        public unsafe bool IsRoot(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_isRoot", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_VertId_isRoot(_Underlying *_this, MR.VertId a);
            return __MR_UnionFind_MR_VertId_isRoot(_UnderlyingPtr, a) != 0;
        }

        /// return parent element of this element, which is equal to given element only for set's root
        /// Generated from method `MR::UnionFind<MR::VertId>::parent`.
        public unsafe MR.VertId Parent(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_parent", ExactSpelling = true)]
            extern static MR.VertId __MR_UnionFind_MR_VertId_parent(_Underlying *_this, MR.VertId a);
            return __MR_UnionFind_MR_VertId_parent(_UnderlyingPtr, a);
        }

        /// gets the parents of all elements as is
        /// Generated from method `MR::UnionFind<MR::VertId>::parents`.
        public unsafe MR.Const_VertMap Parents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_parents", ExactSpelling = true)]
            extern static MR.Const_VertMap._Underlying *__MR_UnionFind_MR_VertId_parents(_Underlying *_this);
            return new(__MR_UnionFind_MR_VertId_parents(_UnderlyingPtr), is_owning: false);
        }
    }

    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::VertId>`.
    /// This is the non-const half of the class.
    public class UnionFind_MRVertId : Const_UnionFind_MRVertId
    {
        internal unsafe UnionFind_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UnionFind_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::VertId>::UnionFind`.
        public unsafe UnionFind_MRVertId(MR._ByValue_UnionFind_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::VertId>::UnionFind`.
        public unsafe UnionFind_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_VertId_Construct(size);
        }

        /// Generated from method `MR::UnionFind<MR::VertId>::operator=`.
        public unsafe MR.UnionFind_MRVertId Assign(MR._ByValue_UnionFind_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_UnionFind_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRVertId._Underlying *_other);
            return new(__MR_UnionFind_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// resets union-find to represent given number of elements, each element is the only one in its disjoint set
        /// Generated from method `MR::UnionFind<MR::VertId>::reset`.
        public unsafe void Reset(ulong size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_reset", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_VertId_reset(_Underlying *_this, ulong size);
            __MR_UnionFind_MR_VertId_reset(_UnderlyingPtr, size);
        }

        /// unite two elements,
        /// \return first: new common root, second: true = union was done, false = first and second were already united
        /// Generated from method `MR::UnionFind<MR::VertId>::unite`.
        public unsafe MR.Std.Pair_MRVertId_Bool Unite(MR.VertId first, MR.VertId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_unite", ExactSpelling = true)]
            extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_UnionFind_MR_VertId_unite(_Underlying *_this, MR.VertId first, MR.VertId second);
            return new(__MR_UnionFind_MR_VertId_unite(_UnderlyingPtr, first, second), is_owning: true);
        }

        /// returns true if given two elements are from one set
        /// Generated from method `MR::UnionFind<MR::VertId>::united`.
        public unsafe bool United(MR.VertId first, MR.VertId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_united", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_VertId_united(_Underlying *_this, MR.VertId first, MR.VertId second);
            return __MR_UnionFind_MR_VertId_united(_UnderlyingPtr, first, second) != 0;
        }

        /// finds the root of the set containing given element with optimizing data structure updates
        /// Generated from method `MR::UnionFind<MR::VertId>::find`.
        public unsafe MR.VertId Find(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_find", ExactSpelling = true)]
            extern static MR.VertId __MR_UnionFind_MR_VertId_find(_Underlying *_this, MR.VertId a);
            return __MR_UnionFind_MR_VertId_find(_UnderlyingPtr, a);
        }

        /// finds the root of the set containing given element with optimizing data structure in the range [begin, end)
        /// Generated from method `MR::UnionFind<MR::VertId>::findUpdateRange`.
        public unsafe MR.VertId FindUpdateRange(MR.VertId a, MR.VertId begin, MR.VertId end)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_findUpdateRange", ExactSpelling = true)]
            extern static MR.VertId __MR_UnionFind_MR_VertId_findUpdateRange(_Underlying *_this, MR.VertId a, MR.VertId begin, MR.VertId end);
            return __MR_UnionFind_MR_VertId_findUpdateRange(_UnderlyingPtr, a, begin, end);
        }

        /// sets the root of corresponding set as the parent of each element, then returns the vector
        /// Generated from method `MR::UnionFind<MR::VertId>::roots`.
        public unsafe MR.Const_VertMap Roots()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_roots", ExactSpelling = true)]
            extern static MR.Const_VertMap._Underlying *__MR_UnionFind_MR_VertId_roots(_Underlying *_this);
            return new(__MR_UnionFind_MR_VertId_roots(_UnderlyingPtr), is_owning: false);
        }

        /// returns the number of elements in the set containing given element
        /// Generated from method `MR::UnionFind<MR::VertId>::sizeOfComp`.
        public unsafe int SizeOfComp(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_VertId_sizeOfComp", ExactSpelling = true)]
            extern static int __MR_UnionFind_MR_VertId_sizeOfComp(_Underlying *_this, MR.VertId a);
            return __MR_UnionFind_MR_VertId_sizeOfComp(_UnderlyingPtr, a);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UnionFind_MRVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UnionFind_MRVertId`/`Const_UnionFind_MRVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UnionFind_MRVertId
    {
        internal readonly Const_UnionFind_MRVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UnionFind_MRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UnionFind_MRVertId(Const_UnionFind_MRVertId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UnionFind_MRVertId(Const_UnionFind_MRVertId arg) {return new(arg);}
        public _ByValue_UnionFind_MRVertId(MR.Misc._Moved<UnionFind_MRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UnionFind_MRVertId(MR.Misc._Moved<UnionFind_MRVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UnionFind_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UnionFind_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRVertId`/`Const_UnionFind_MRVertId` directly.
    public class _InOptMut_UnionFind_MRVertId
    {
        public UnionFind_MRVertId? Opt;

        public _InOptMut_UnionFind_MRVertId() {}
        public _InOptMut_UnionFind_MRVertId(UnionFind_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_UnionFind_MRVertId(UnionFind_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `UnionFind_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UnionFind_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRVertId`/`Const_UnionFind_MRVertId` to pass it to the function.
    public class _InOptConst_UnionFind_MRVertId
    {
        public Const_UnionFind_MRVertId? Opt;

        public _InOptConst_UnionFind_MRVertId() {}
        public _InOptConst_UnionFind_MRVertId(Const_UnionFind_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_UnionFind_MRVertId(Const_UnionFind_MRVertId value) {return new(value);}
    }

    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_UnionFind_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UnionFind_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_UnionFind_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UnionFind_MRUndirectedEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UnionFind_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::UndirectedEdgeId>::UnionFind`.
        public unsafe Const_UnionFind_MRUndirectedEdgeId(MR._ByValue_UnionFind_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::UndirectedEdgeId>::UnionFind`.
        public unsafe Const_UnionFind_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_Construct(size);
        }

        /// returns the number of elements in union-find
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_UnionFind_MR_UndirectedEdgeId_size(_Underlying *_this);
            return __MR_UnionFind_MR_UndirectedEdgeId_size(_UnderlyingPtr);
        }

        /// returns true if given element is the root of some set
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::isRoot`.
        public unsafe bool IsRoot(MR.UndirectedEdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_isRoot", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_UndirectedEdgeId_isRoot(_Underlying *_this, MR.UndirectedEdgeId a);
            return __MR_UnionFind_MR_UndirectedEdgeId_isRoot(_UnderlyingPtr, a) != 0;
        }

        /// return parent element of this element, which is equal to given element only for set's root
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::parent`.
        public unsafe MR.UndirectedEdgeId Parent(MR.UndirectedEdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_parent", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UnionFind_MR_UndirectedEdgeId_parent(_Underlying *_this, MR.UndirectedEdgeId a);
            return __MR_UnionFind_MR_UndirectedEdgeId_parent(_UnderlyingPtr, a);
        }

        /// gets the parents of all elements as is
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::parents`.
        public unsafe MR.Const_UndirectedEdgeMap Parents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_parents", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeMap._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_parents(_Underlying *_this);
            return new(__MR_UnionFind_MR_UndirectedEdgeId_parents(_UnderlyingPtr), is_owning: false);
        }
    }

    /** 
    * \brief Union-find data structure for representing disjoin sets of elements with few very quick operations:
    * 1) union of two sets in one,
    * 2) checking whether two elements pertain to the same set,
    * 3) finding representative element (root) of each set by any set's element
    * \tparam I is the identifier of a set's element, e.g. FaceId
    *
    */
    /// Generated from class `MR::UnionFind<MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class UnionFind_MRUndirectedEdgeId : Const_UnionFind_MRUndirectedEdgeId
    {
        internal unsafe UnionFind_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UnionFind_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UnionFind<MR::UndirectedEdgeId>::UnionFind`.
        public unsafe UnionFind_MRUndirectedEdgeId(MR._ByValue_UnionFind_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates union-find with given number of elements, each element is the only one in its disjoint set
        /// Generated from constructor `MR::UnionFind<MR::UndirectedEdgeId>::UnionFind`.
        public unsafe UnionFind_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_UnionFind_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.UnionFind_MRUndirectedEdgeId Assign(MR._ByValue_UnionFind_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UnionFind_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_UnionFind_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// resets union-find to represent given number of elements, each element is the only one in its disjoint set
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::reset`.
        public unsafe void Reset(ulong size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_reset", ExactSpelling = true)]
            extern static void __MR_UnionFind_MR_UndirectedEdgeId_reset(_Underlying *_this, ulong size);
            __MR_UnionFind_MR_UndirectedEdgeId_reset(_UnderlyingPtr, size);
        }

        /// unite two elements,
        /// \return first: new common root, second: true = union was done, false = first and second were already united
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::unite`.
        public unsafe MR.Std.Pair_MRUndirectedEdgeId_Bool Unite(MR.UndirectedEdgeId first, MR.UndirectedEdgeId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_unite", ExactSpelling = true)]
            extern static MR.Std.Pair_MRUndirectedEdgeId_Bool._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_unite(_Underlying *_this, MR.UndirectedEdgeId first, MR.UndirectedEdgeId second);
            return new(__MR_UnionFind_MR_UndirectedEdgeId_unite(_UnderlyingPtr, first, second), is_owning: true);
        }

        /// returns true if given two elements are from one set
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::united`.
        public unsafe bool United(MR.UndirectedEdgeId first, MR.UndirectedEdgeId second)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_united", ExactSpelling = true)]
            extern static byte __MR_UnionFind_MR_UndirectedEdgeId_united(_Underlying *_this, MR.UndirectedEdgeId first, MR.UndirectedEdgeId second);
            return __MR_UnionFind_MR_UndirectedEdgeId_united(_UnderlyingPtr, first, second) != 0;
        }

        /// finds the root of the set containing given element with optimizing data structure updates
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::find`.
        public unsafe MR.UndirectedEdgeId Find(MR.UndirectedEdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_find", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UnionFind_MR_UndirectedEdgeId_find(_Underlying *_this, MR.UndirectedEdgeId a);
            return __MR_UnionFind_MR_UndirectedEdgeId_find(_UnderlyingPtr, a);
        }

        /// finds the root of the set containing given element with optimizing data structure in the range [begin, end)
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::findUpdateRange`.
        public unsafe MR.UndirectedEdgeId FindUpdateRange(MR.UndirectedEdgeId a, MR.UndirectedEdgeId begin, MR.UndirectedEdgeId end)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_findUpdateRange", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UnionFind_MR_UndirectedEdgeId_findUpdateRange(_Underlying *_this, MR.UndirectedEdgeId a, MR.UndirectedEdgeId begin, MR.UndirectedEdgeId end);
            return __MR_UnionFind_MR_UndirectedEdgeId_findUpdateRange(_UnderlyingPtr, a, begin, end);
        }

        /// sets the root of corresponding set as the parent of each element, then returns the vector
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::roots`.
        public unsafe MR.Const_UndirectedEdgeMap Roots()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_roots", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeMap._Underlying *__MR_UnionFind_MR_UndirectedEdgeId_roots(_Underlying *_this);
            return new(__MR_UnionFind_MR_UndirectedEdgeId_roots(_UnderlyingPtr), is_owning: false);
        }

        /// returns the number of elements in the set containing given element
        /// Generated from method `MR::UnionFind<MR::UndirectedEdgeId>::sizeOfComp`.
        public unsafe int SizeOfComp(MR.UndirectedEdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnionFind_MR_UndirectedEdgeId_sizeOfComp", ExactSpelling = true)]
            extern static int __MR_UnionFind_MR_UndirectedEdgeId_sizeOfComp(_Underlying *_this, MR.UndirectedEdgeId a);
            return __MR_UnionFind_MR_UndirectedEdgeId_sizeOfComp(_UnderlyingPtr, a);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UnionFind_MRUndirectedEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UnionFind_MRUndirectedEdgeId`/`Const_UnionFind_MRUndirectedEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UnionFind_MRUndirectedEdgeId
    {
        internal readonly Const_UnionFind_MRUndirectedEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UnionFind_MRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UnionFind_MRUndirectedEdgeId(Const_UnionFind_MRUndirectedEdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UnionFind_MRUndirectedEdgeId(Const_UnionFind_MRUndirectedEdgeId arg) {return new(arg);}
        public _ByValue_UnionFind_MRUndirectedEdgeId(MR.Misc._Moved<UnionFind_MRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UnionFind_MRUndirectedEdgeId(MR.Misc._Moved<UnionFind_MRUndirectedEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UnionFind_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UnionFind_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRUndirectedEdgeId`/`Const_UnionFind_MRUndirectedEdgeId` directly.
    public class _InOptMut_UnionFind_MRUndirectedEdgeId
    {
        public UnionFind_MRUndirectedEdgeId? Opt;

        public _InOptMut_UnionFind_MRUndirectedEdgeId() {}
        public _InOptMut_UnionFind_MRUndirectedEdgeId(UnionFind_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_UnionFind_MRUndirectedEdgeId(UnionFind_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `UnionFind_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UnionFind_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnionFind_MRUndirectedEdgeId`/`Const_UnionFind_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_UnionFind_MRUndirectedEdgeId
    {
        public Const_UnionFind_MRUndirectedEdgeId? Opt;

        public _InOptConst_UnionFind_MRUndirectedEdgeId() {}
        public _InOptConst_UnionFind_MRUndirectedEdgeId(Const_UnionFind_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_UnionFind_MRUndirectedEdgeId(Const_UnionFind_MRUndirectedEdgeId value) {return new(value);}
    }
}

public static partial class MR
{
    public enum OutEdge2 : sbyte
    {
        Invalid = -1,
        PlusY = 0,
        MinusY = 1,
        PlusX = 2,
        MinusX = 3,
        Count = 4,
    }

    /// a class for converting 2D integer coordinates into 1D linear coordinates and backward
    /// Generated from class `MR::RectIndexer`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceMap`
    ///     `MR::Matrix<float>`
    /// This is the const half of the class.
    public class Const_RectIndexer : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_RectIndexer_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_RectIndexer_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_RectIndexer_UseCount();
                return __MR_std_shared_ptr_MR_RectIndexer_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RectIndexer_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_RectIndexer_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_RectIndexer(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RectIndexer_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RectIndexer_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RectIndexer_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RectIndexer_ConstructNonOwning(ptr);
        }

        internal unsafe Const_RectIndexer(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe RectIndexer _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RectIndexer_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_RectIndexer_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RectIndexer_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RectIndexer_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RectIndexer_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_RectIndexer_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_RectIndexer_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RectIndexer() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RectIndexer() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_DefaultConstruct();
            _LateMakeShared(__MR_RectIndexer_DefaultConstruct());
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public unsafe Const_RectIndexer(MR.Const_RectIndexer _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_ConstructFromAnother(MR.RectIndexer._Underlying *_other);
            _LateMakeShared(__MR_RectIndexer_ConstructFromAnother(_other._UnderlyingPtr));
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public unsafe Const_RectIndexer(MR.Const_Vector2i dims) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_Construct", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_Construct(MR.Const_Vector2i._Underlying *dims);
            _LateMakeShared(__MR_RectIndexer_Construct(dims._UnderlyingPtr));
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public static unsafe implicit operator Const_RectIndexer(MR.Const_Vector2i dims) {return new(dims);}

        /// Generated from method `MR::RectIndexer::dims`.
        public unsafe MR.Const_Vector2i Dims()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_dims", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_RectIndexer_dims(_Underlying *_this);
            return new(__MR_RectIndexer_dims(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RectIndexer::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_size", ExactSpelling = true)]
            extern static ulong __MR_RectIndexer_size(_Underlying *_this);
            return __MR_RectIndexer_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::RectIndexer::toPos`.
        public unsafe MR.Vector2i ToPos(MR.PixelId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_toPos_MR_PixelId", ExactSpelling = true)]
            extern static MR.Vector2i __MR_RectIndexer_toPos_MR_PixelId(_Underlying *_this, MR.PixelId id);
            return __MR_RectIndexer_toPos_MR_PixelId(_UnderlyingPtr, id);
        }

        /// Generated from method `MR::RectIndexer::toPos`.
        public unsafe MR.Vector2i ToPos(ulong id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_toPos_uint64_t", ExactSpelling = true)]
            extern static MR.Vector2i __MR_RectIndexer_toPos_uint64_t(_Underlying *_this, ulong id);
            return __MR_RectIndexer_toPos_uint64_t(_UnderlyingPtr, id);
        }

        /// Generated from method `MR::RectIndexer::toPixelId`.
        public unsafe MR.PixelId ToPixelId(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_toPixelId", ExactSpelling = true)]
            extern static MR.PixelId __MR_RectIndexer_toPixelId(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_RectIndexer_toPixelId(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::RectIndexer::toIndex`.
        public unsafe ulong ToIndex(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_toIndex", ExactSpelling = true)]
            extern static ulong __MR_RectIndexer_toIndex(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_RectIndexer_toIndex(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// returns true if v1 is within at most 4 neighbors of v0
        /// Generated from method `MR::RectIndexer::areNeigbors`.
        public unsafe bool AreNeigbors(MR.PixelId v0, MR.PixelId v1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_areNeigbors_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_RectIndexer_areNeigbors_MR_PixelId(_Underlying *_this, MR.PixelId v0, MR.PixelId v1);
            return __MR_RectIndexer_areNeigbors_MR_PixelId(_UnderlyingPtr, v0, v1) != 0;
        }

        /// Generated from method `MR::RectIndexer::areNeigbors`.
        public unsafe bool AreNeigbors(MR.Const_Vector2i pos0, MR.Const_Vector2i pos1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_areNeigbors_MR_Vector2i", ExactSpelling = true)]
            extern static byte __MR_RectIndexer_areNeigbors_MR_Vector2i(_Underlying *_this, MR.Const_Vector2i._Underlying *pos0, MR.Const_Vector2i._Underlying *pos1);
            return __MR_RectIndexer_areNeigbors_MR_Vector2i(_UnderlyingPtr, pos0._UnderlyingPtr, pos1._UnderlyingPtr) != 0;
        }

        /// returns id of v's neighbor specified by the edge
        /// Generated from method `MR::RectIndexer::getNeighbor`.
        public unsafe MR.PixelId GetNeighbor(MR.PixelId v, MR.OutEdge2 toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_getNeighbor_2", ExactSpelling = true)]
            extern static MR.PixelId __MR_RectIndexer_getNeighbor_2(_Underlying *_this, MR.PixelId v, MR.OutEdge2 toNei);
            return __MR_RectIndexer_getNeighbor_2(_UnderlyingPtr, v, toNei);
        }

        /// Generated from method `MR::RectIndexer::getNeighbor`.
        public unsafe MR.PixelId GetNeighbor(MR.PixelId v, MR.Const_Vector2i pos, MR.OutEdge2 toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_getNeighbor_3", ExactSpelling = true)]
            extern static MR.PixelId __MR_RectIndexer_getNeighbor_3(_Underlying *_this, MR.PixelId v, MR.Const_Vector2i._Underlying *pos, MR.OutEdge2 toNei);
            return __MR_RectIndexer_getNeighbor_3(_UnderlyingPtr, v, pos._UnderlyingPtr, toNei);
        }
    }

    /// a class for converting 2D integer coordinates into 1D linear coordinates and backward
    /// Generated from class `MR::RectIndexer`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceMap`
    ///     `MR::Matrix<float>`
    /// This is the non-const half of the class.
    public class RectIndexer : Const_RectIndexer
    {
        internal unsafe RectIndexer(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe RectIndexer(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe RectIndexer() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_DefaultConstruct();
            _LateMakeShared(__MR_RectIndexer_DefaultConstruct());
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public unsafe RectIndexer(MR.Const_RectIndexer _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_ConstructFromAnother(MR.RectIndexer._Underlying *_other);
            _LateMakeShared(__MR_RectIndexer_ConstructFromAnother(_other._UnderlyingPtr));
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public unsafe RectIndexer(MR.Const_Vector2i dims) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_Construct", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_Construct(MR.Const_Vector2i._Underlying *dims);
            _LateMakeShared(__MR_RectIndexer_Construct(dims._UnderlyingPtr));
        }

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public static unsafe implicit operator RectIndexer(MR.Const_Vector2i dims) {return new(dims);}

        /// Generated from method `MR::RectIndexer::operator=`.
        public unsafe MR.RectIndexer Assign(MR.Const_RectIndexer _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_RectIndexer_AssignFromAnother(_Underlying *_this, MR.RectIndexer._Underlying *_other);
            return new(__MR_RectIndexer_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RectIndexer::resize`.
        public unsafe void Resize(MR.Const_Vector2i dims)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RectIndexer_resize", ExactSpelling = true)]
            extern static void __MR_RectIndexer_resize(_Underlying *_this, MR.Const_Vector2i._Underlying *dims);
            __MR_RectIndexer_resize(_UnderlyingPtr, dims._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RectIndexer` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RectIndexer`/`Const_RectIndexer` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RectIndexer
    {
        internal readonly Const_RectIndexer? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RectIndexer() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RectIndexer(Const_RectIndexer new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RectIndexer(Const_RectIndexer arg) {return new(arg);}
        public _ByValue_RectIndexer(MR.Misc._Moved<RectIndexer> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RectIndexer(MR.Misc._Moved<RectIndexer> arg) {return new(arg);}

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public static unsafe implicit operator _ByValue_RectIndexer(MR.Const_Vector2i dims) {return new MR.RectIndexer(dims);}
    }

    /// This is used for optional parameters of class `RectIndexer` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RectIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RectIndexer`/`Const_RectIndexer` directly.
    public class _InOptMut_RectIndexer
    {
        public RectIndexer? Opt;

        public _InOptMut_RectIndexer() {}
        public _InOptMut_RectIndexer(RectIndexer value) {Opt = value;}
        public static implicit operator _InOptMut_RectIndexer(RectIndexer value) {return new(value);}
    }

    /// This is used for optional parameters of class `RectIndexer` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RectIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RectIndexer`/`Const_RectIndexer` to pass it to the function.
    public class _InOptConst_RectIndexer
    {
        public Const_RectIndexer? Opt;

        public _InOptConst_RectIndexer() {}
        public _InOptConst_RectIndexer(Const_RectIndexer value) {Opt = value;}
        public static implicit operator _InOptConst_RectIndexer(Const_RectIndexer value) {return new(value);}

        /// Generated from constructor `MR::RectIndexer::RectIndexer`.
        public static unsafe implicit operator _InOptConst_RectIndexer(MR.Const_Vector2i dims) {return new MR.RectIndexer(dims);}
    }

    /// Generated from function `MR::opposite`.
    public static MR.OutEdge2 Opposite(MR.OutEdge2 e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_opposite_MR_OutEdge2", ExactSpelling = true)]
        extern static MR.OutEdge2 __MR_opposite_MR_OutEdge2(MR.OutEdge2 e);
        return __MR_opposite_MR_OutEdge2(e);
    }

    /// expands PixelBitSet with given number of steps
    /// Generated from function `MR::expandPixelMask`.
    /// Parameter `expansion` defaults to `1`.
    public static unsafe void ExpandPixelMask(MR.PixelBitSet mask, MR.Const_RectIndexer indexer, int? expansion = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expandPixelMask", ExactSpelling = true)]
        extern static void __MR_expandPixelMask(MR.PixelBitSet._Underlying *mask, MR.Const_RectIndexer._Underlying *indexer, int *expansion);
        int __deref_expansion = expansion.GetValueOrDefault();
        __MR_expandPixelMask(mask._UnderlyingPtr, indexer._UnderlyingPtr, expansion.HasValue ? &__deref_expansion : null);
    }

    /// shrinks PixelBitSet with given number of steps
    /// Generated from function `MR::shrinkPixelMask`.
    /// Parameter `shrinkage` defaults to `1`.
    public static unsafe void ShrinkPixelMask(MR.PixelBitSet mask, MR.Const_RectIndexer indexer, int? shrinkage = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_shrinkPixelMask", ExactSpelling = true)]
        extern static void __MR_shrinkPixelMask(MR.PixelBitSet._Underlying *mask, MR.Const_RectIndexer._Underlying *indexer, int *shrinkage);
        int __deref_shrinkage = shrinkage.GetValueOrDefault();
        __MR_shrinkPixelMask(mask._UnderlyingPtr, indexer._UnderlyingPtr, shrinkage.HasValue ? &__deref_shrinkage : null);
    }
}

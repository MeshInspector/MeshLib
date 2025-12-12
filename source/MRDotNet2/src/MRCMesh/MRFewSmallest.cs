public static partial class MR
{
    /// the class stores some number of smallest elements from a larger number of candidates
    /// Generated from class `MR::FewSmallest<MR::PointsProjectionResult>`.
    /// This is the const half of the class.
    public class Const_FewSmallest_MRPointsProjectionResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FewSmallest_MRPointsProjectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_Destroy", ExactSpelling = true)]
            extern static void __MR_FewSmallest_MR_PointsProjectionResult_Destroy(_Underlying *_this);
            __MR_FewSmallest_MR_PointsProjectionResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FewSmallest_MRPointsProjectionResult() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FewSmallest_MRPointsProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct();
        }

        /// Generated from constructor `MR::FewSmallest<MR::PointsProjectionResult>::FewSmallest`.
        public unsafe Const_FewSmallest_MRPointsProjectionResult(MR._ByValue_FewSmallest_MRPointsProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FewSmallest_MRPointsProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// configure the object to store at most given number of elements
        /// Generated from constructor `MR::FewSmallest<MR::PointsProjectionResult>::FewSmallest`.
        public unsafe Const_FewSmallest_MRPointsProjectionResult(ulong maxElms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_Construct", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_Construct(ulong maxElms);
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_Construct(maxElms);
        }

        /// returns the maximum number of elements to be stored here
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::maxElms`.
        public unsafe ulong MaxElms()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_maxElms", ExactSpelling = true)]
            extern static ulong __MR_FewSmallest_MR_PointsProjectionResult_maxElms(_Underlying *_this);
            return __MR_FewSmallest_MR_PointsProjectionResult_maxElms(_UnderlyingPtr);
        }

        /// returns whether the container is currently empty
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_empty", ExactSpelling = true)]
            extern static byte __MR_FewSmallest_MR_PointsProjectionResult_empty(_Underlying *_this);
            return __MR_FewSmallest_MR_PointsProjectionResult_empty(_UnderlyingPtr) != 0;
        }

        /// returns current number of stored element
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_size", ExactSpelling = true)]
            extern static ulong __MR_FewSmallest_MR_PointsProjectionResult_size(_Underlying *_this);
            return __MR_FewSmallest_MR_PointsProjectionResult_size(_UnderlyingPtr);
        }

        /// returns whether we have already maximum number of elements stored
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::full`.
        public unsafe bool Full()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_full", ExactSpelling = true)]
            extern static byte __MR_FewSmallest_MR_PointsProjectionResult_full(_Underlying *_this);
            return __MR_FewSmallest_MR_PointsProjectionResult_full(_UnderlyingPtr) != 0;
        }

        /// returns the smallest elements found so far
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::get`.
        public unsafe MR.Std.Const_Vector_MRPointsProjectionResult Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_get", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_get(_Underlying *_this);
            return new(__MR_FewSmallest_MR_PointsProjectionResult_get(_UnderlyingPtr), is_owning: false);
        }

        /// returns the largest among stored smallest elements
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::top`.
        public unsafe MR.Const_PointsProjectionResult Top()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_top", ExactSpelling = true)]
            extern static MR.Const_PointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_top(_Underlying *_this);
            return new(__MR_FewSmallest_MR_PointsProjectionResult_top(_UnderlyingPtr), is_owning: false);
        }

        /// returns the largest among stored smallest elements or given element if this is empty
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::topOr`.
        public unsafe MR.Const_PointsProjectionResult TopOr(MR.Const_PointsProjectionResult emptyRes)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_topOr", ExactSpelling = true)]
            extern static MR.Const_PointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_topOr(_Underlying *_this, MR.Const_PointsProjectionResult._Underlying *emptyRes);
            return new(__MR_FewSmallest_MR_PointsProjectionResult_topOr(_UnderlyingPtr, emptyRes._UnderlyingPtr), is_owning: false);
        }
    }

    /// the class stores some number of smallest elements from a larger number of candidates
    /// Generated from class `MR::FewSmallest<MR::PointsProjectionResult>`.
    /// This is the non-const half of the class.
    public class FewSmallest_MRPointsProjectionResult : Const_FewSmallest_MRPointsProjectionResult
    {
        internal unsafe FewSmallest_MRPointsProjectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe FewSmallest_MRPointsProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_DefaultConstruct();
        }

        /// Generated from constructor `MR::FewSmallest<MR::PointsProjectionResult>::FewSmallest`.
        public unsafe FewSmallest_MRPointsProjectionResult(MR._ByValue_FewSmallest_MRPointsProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FewSmallest_MRPointsProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// configure the object to store at most given number of elements
        /// Generated from constructor `MR::FewSmallest<MR::PointsProjectionResult>::FewSmallest`.
        public unsafe FewSmallest_MRPointsProjectionResult(ulong maxElms) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_Construct", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_Construct(ulong maxElms);
            _UnderlyingPtr = __MR_FewSmallest_MR_PointsProjectionResult_Construct(maxElms);
        }

        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::operator=`.
        public unsafe MR.FewSmallest_MRPointsProjectionResult Assign(MR._ByValue_FewSmallest_MRPointsProjectionResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_FewSmallest_MR_PointsProjectionResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FewSmallest_MRPointsProjectionResult._Underlying *_other);
            return new(__MR_FewSmallest_MR_PointsProjectionResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// clears the content and reconfigure the object to store at most given number of elements
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::reset`.
        public unsafe void Reset(ulong maxElms)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_reset", ExactSpelling = true)]
            extern static void __MR_FewSmallest_MR_PointsProjectionResult_reset(_Underlying *_this, ulong maxElms);
            __MR_FewSmallest_MR_PointsProjectionResult_reset(_UnderlyingPtr, maxElms);
        }

        /// considers one more element, storing it if it is within the smallest
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::push`.
        public unsafe void Push(MR.Const_PointsProjectionResult t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_push", ExactSpelling = true)]
            extern static void __MR_FewSmallest_MR_PointsProjectionResult_push(_Underlying *_this, MR.PointsProjectionResult._Underlying *t);
            __MR_FewSmallest_MR_PointsProjectionResult_push(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// removes all stored elements
        /// Generated from method `MR::FewSmallest<MR::PointsProjectionResult>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FewSmallest_MR_PointsProjectionResult_clear", ExactSpelling = true)]
            extern static void __MR_FewSmallest_MR_PointsProjectionResult_clear(_Underlying *_this);
            __MR_FewSmallest_MR_PointsProjectionResult_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FewSmallest_MRPointsProjectionResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FewSmallest_MRPointsProjectionResult`/`Const_FewSmallest_MRPointsProjectionResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FewSmallest_MRPointsProjectionResult
    {
        internal readonly Const_FewSmallest_MRPointsProjectionResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FewSmallest_MRPointsProjectionResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FewSmallest_MRPointsProjectionResult(Const_FewSmallest_MRPointsProjectionResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FewSmallest_MRPointsProjectionResult(Const_FewSmallest_MRPointsProjectionResult arg) {return new(arg);}
        public _ByValue_FewSmallest_MRPointsProjectionResult(MR.Misc._Moved<FewSmallest_MRPointsProjectionResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FewSmallest_MRPointsProjectionResult(MR.Misc._Moved<FewSmallest_MRPointsProjectionResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FewSmallest_MRPointsProjectionResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FewSmallest_MRPointsProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FewSmallest_MRPointsProjectionResult`/`Const_FewSmallest_MRPointsProjectionResult` directly.
    public class _InOptMut_FewSmallest_MRPointsProjectionResult
    {
        public FewSmallest_MRPointsProjectionResult? Opt;

        public _InOptMut_FewSmallest_MRPointsProjectionResult() {}
        public _InOptMut_FewSmallest_MRPointsProjectionResult(FewSmallest_MRPointsProjectionResult value) {Opt = value;}
        public static implicit operator _InOptMut_FewSmallest_MRPointsProjectionResult(FewSmallest_MRPointsProjectionResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `FewSmallest_MRPointsProjectionResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FewSmallest_MRPointsProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FewSmallest_MRPointsProjectionResult`/`Const_FewSmallest_MRPointsProjectionResult` to pass it to the function.
    public class _InOptConst_FewSmallest_MRPointsProjectionResult
    {
        public Const_FewSmallest_MRPointsProjectionResult? Opt;

        public _InOptConst_FewSmallest_MRPointsProjectionResult() {}
        public _InOptConst_FewSmallest_MRPointsProjectionResult(Const_FewSmallest_MRPointsProjectionResult value) {Opt = value;}
        public static implicit operator _InOptConst_FewSmallest_MRPointsProjectionResult(Const_FewSmallest_MRPointsProjectionResult value) {return new(value);}
    }
}

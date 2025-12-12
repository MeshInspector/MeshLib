public static partial class MR
{
    /// minimum and maximum of some elements of type T,
    /// and the indices of where minimum and maximum are reached of type I
    /// Generated from class `MR::MinMaxArg<float, MR::VertId>`.
    /// This is the const half of the class.
    public class Const_MinMaxArg_Float_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MinMaxArg_Float_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_MinMaxArg_float_MR_VertId_Destroy(_Underlying *_this);
            __MR_MinMaxArg_float_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MinMaxArg_Float_MRVertId() {Dispose(false);}

        public unsafe float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_Get_min", ExactSpelling = true)]
                extern static float *__MR_MinMaxArg_float_MR_VertId_Get_min(_Underlying *_this);
                return *__MR_MinMaxArg_float_MR_VertId_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_Get_max", ExactSpelling = true)]
                extern static float *__MR_MinMaxArg_float_MR_VertId_Get_max(_Underlying *_this);
                return *__MR_MinMaxArg_float_MR_VertId_Get_max(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_VertId MinArg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_Get_minArg", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_MinMaxArg_float_MR_VertId_Get_minArg(_Underlying *_this);
                return new(__MR_MinMaxArg_float_MR_VertId_Get_minArg(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertId MaxArg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_Get_maxArg", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_MinMaxArg_float_MR_VertId_Get_maxArg(_Underlying *_this);
                return new(__MR_MinMaxArg_float_MR_VertId_Get_maxArg(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MinMaxArg_Float_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_DefaultConstruct();
        }

        /// Constructs `MR::MinMaxArg<float, MR::VertId>` elementwise.
        public unsafe Const_MinMaxArg_Float_MRVertId(float min, float max, MR.VertId minArg, MR.VertId maxArg) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_ConstructFrom(float min, float max, MR.VertId minArg, MR.VertId maxArg);
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_ConstructFrom(min, max, minArg, maxArg);
        }

        /// Generated from constructor `MR::MinMaxArg<float, MR::VertId>::MinMaxArg`.
        public unsafe Const_MinMaxArg_Float_MRVertId(MR.Const_MinMaxArg_Float_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_ConstructFromAnother(MR.MinMaxArg_Float_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::minPair`.
        public unsafe MR.Std.Pair_Float_MRVertId MinPair()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_minPair", ExactSpelling = true)]
            extern static MR.Std.Pair_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_minPair(_Underlying *_this);
            return new(__MR_MinMaxArg_float_MR_VertId_minPair(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::maxPair`.
        public unsafe MR.Std.Pair_Float_MRVertId MaxPair()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_maxPair", ExactSpelling = true)]
            extern static MR.Std.Pair_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_maxPair(_Underlying *_this);
            return new(__MR_MinMaxArg_float_MR_VertId_maxPair(_UnderlyingPtr), is_owning: true);
        }
    }

    /// minimum and maximum of some elements of type T,
    /// and the indices of where minimum and maximum are reached of type I
    /// Generated from class `MR::MinMaxArg<float, MR::VertId>`.
    /// This is the non-const half of the class.
    public class MinMaxArg_Float_MRVertId : Const_MinMaxArg_Float_MRVertId
    {
        internal unsafe MinMaxArg_Float_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_GetMutable_min", ExactSpelling = true)]
                extern static float *__MR_MinMaxArg_float_MR_VertId_GetMutable_min(_Underlying *_this);
                return ref *__MR_MinMaxArg_float_MR_VertId_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_GetMutable_max", ExactSpelling = true)]
                extern static float *__MR_MinMaxArg_float_MR_VertId_GetMutable_max(_Underlying *_this);
                return ref *__MR_MinMaxArg_float_MR_VertId_GetMutable_max(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Mut_VertId MinArg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_GetMutable_minArg", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_MinMaxArg_float_MR_VertId_GetMutable_minArg(_Underlying *_this);
                return new(__MR_MinMaxArg_float_MR_VertId_GetMutable_minArg(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_VertId MaxArg
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_GetMutable_maxArg", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_MinMaxArg_float_MR_VertId_GetMutable_maxArg(_Underlying *_this);
                return new(__MR_MinMaxArg_float_MR_VertId_GetMutable_maxArg(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MinMaxArg_Float_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_DefaultConstruct();
        }

        /// Constructs `MR::MinMaxArg<float, MR::VertId>` elementwise.
        public unsafe MinMaxArg_Float_MRVertId(float min, float max, MR.VertId minArg, MR.VertId maxArg) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_ConstructFrom(float min, float max, MR.VertId minArg, MR.VertId maxArg);
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_ConstructFrom(min, max, minArg, maxArg);
        }

        /// Generated from constructor `MR::MinMaxArg<float, MR::VertId>::MinMaxArg`.
        public unsafe MinMaxArg_Float_MRVertId(MR.Const_MinMaxArg_Float_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_ConstructFromAnother(MR.MinMaxArg_Float_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_MinMaxArg_float_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::operator=`.
        public unsafe MR.MinMaxArg_Float_MRVertId Assign(MR.Const_MinMaxArg_Float_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_MinMaxArg_float_MR_VertId_AssignFromAnother(_Underlying *_this, MR.MinMaxArg_Float_MRVertId._Underlying *_other);
            return new(__MR_MinMaxArg_float_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// changes min(Arg) and max(Arg) if necessary to include given point
        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::include`.
        public unsafe void Include(MR.Std.Const_Pair_Float_MRVertId p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_include_1_std_pair_float_MR_VertId", ExactSpelling = true)]
            extern static void __MR_MinMaxArg_float_MR_VertId_include_1_std_pair_float_MR_VertId(_Underlying *_this, MR.Std.Const_Pair_Float_MRVertId._Underlying *p);
            __MR_MinMaxArg_float_MR_VertId_include_1_std_pair_float_MR_VertId(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// changes min(Arg) and max(Arg) if necessary to include given point
        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::include`.
        public unsafe void Include(float v, MR.VertId arg)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_include_2", ExactSpelling = true)]
            extern static void __MR_MinMaxArg_float_MR_VertId_include_2(_Underlying *_this, float v, MR.VertId arg);
            __MR_MinMaxArg_float_MR_VertId_include_2(_UnderlyingPtr, v, arg);
        }

        /// changes min(Arg) and max(Arg) if necessary to include given segment
        /// Generated from method `MR::MinMaxArg<float, MR::VertId>::include`.
        public unsafe void Include(MR.Const_MinMaxArg_Float_MRVertId s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MinMaxArg_float_MR_VertId_include_1_MR_MinMaxArg_float_MR_VertId", ExactSpelling = true)]
            extern static void __MR_MinMaxArg_float_MR_VertId_include_1_MR_MinMaxArg_float_MR_VertId(_Underlying *_this, MR.Const_MinMaxArg_Float_MRVertId._Underlying *s);
            __MR_MinMaxArg_float_MR_VertId_include_1_MR_MinMaxArg_float_MR_VertId(_UnderlyingPtr, s._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `MinMaxArg_Float_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MinMaxArg_Float_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MinMaxArg_Float_MRVertId`/`Const_MinMaxArg_Float_MRVertId` directly.
    public class _InOptMut_MinMaxArg_Float_MRVertId
    {
        public MinMaxArg_Float_MRVertId? Opt;

        public _InOptMut_MinMaxArg_Float_MRVertId() {}
        public _InOptMut_MinMaxArg_Float_MRVertId(MinMaxArg_Float_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_MinMaxArg_Float_MRVertId(MinMaxArg_Float_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `MinMaxArg_Float_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MinMaxArg_Float_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MinMaxArg_Float_MRVertId`/`Const_MinMaxArg_Float_MRVertId` to pass it to the function.
    public class _InOptConst_MinMaxArg_Float_MRVertId
    {
        public Const_MinMaxArg_Float_MRVertId? Opt;

        public _InOptConst_MinMaxArg_Float_MRVertId() {}
        public _InOptConst_MinMaxArg_Float_MRVertId(Const_MinMaxArg_Float_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_MinMaxArg_Float_MRVertId(Const_MinMaxArg_Float_MRVertId value) {return new(value);}
    }
}

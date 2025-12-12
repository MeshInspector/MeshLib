public static partial class MR
{
    /// use this adapter to call functions expecting PartMapping parameter to receive src2tgt dense maps
    /// Generated from class `MR::Src2TgtMaps`.
    /// This is the const half of the class.
    public class Const_Src2TgtMaps : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Src2TgtMaps(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_Destroy", ExactSpelling = true)]
            extern static void __MR_Src2TgtMaps_Destroy(_Underlying *_this);
            __MR_Src2TgtMaps_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Src2TgtMaps() {Dispose(false);}

        /// Generated from constructor `MR::Src2TgtMaps::Src2TgtMaps`.
        public unsafe Const_Src2TgtMaps(MR._ByValue_Src2TgtMaps _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Src2TgtMaps._Underlying *__MR_Src2TgtMaps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Src2TgtMaps._Underlying *_other);
            _UnderlyingPtr = __MR_Src2TgtMaps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Src2TgtMaps::Src2TgtMaps`.
        public unsafe Const_Src2TgtMaps(MR.FaceMap? outFmap, MR.VertMap? outVmap, MR.WholeEdgeMap? outEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_Construct", ExactSpelling = true)]
            extern static MR.Src2TgtMaps._Underlying *__MR_Src2TgtMaps_Construct(MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            _UnderlyingPtr = __MR_Src2TgtMaps_Construct(outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// Generated from conversion operator `MR::Src2TgtMaps::operator const MR::PartMapping &`.
        public static unsafe implicit operator MR.Const_PartMapping(MR.Const_Src2TgtMaps _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_ConvertTo_const_MR_PartMapping_ref", ExactSpelling = true)]
            extern static MR.Const_PartMapping._Underlying *__MR_Src2TgtMaps_ConvertTo_const_MR_PartMapping_ref(MR.Const_Src2TgtMaps._Underlying *_this);
            return new(__MR_Src2TgtMaps_ConvertTo_const_MR_PartMapping_ref(_this._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Src2TgtMaps::getPartMapping`.
        public unsafe MR.Const_PartMapping GetPartMapping()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_getPartMapping", ExactSpelling = true)]
            extern static MR.Const_PartMapping._Underlying *__MR_Src2TgtMaps_getPartMapping(_Underlying *_this);
            return new(__MR_Src2TgtMaps_getPartMapping(_UnderlyingPtr), is_owning: false);
        }
    }

    /// use this adapter to call functions expecting PartMapping parameter to receive src2tgt dense maps
    /// Generated from class `MR::Src2TgtMaps`.
    /// This is the non-const half of the class.
    public class Src2TgtMaps : Const_Src2TgtMaps
    {
        internal unsafe Src2TgtMaps(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::Src2TgtMaps::Src2TgtMaps`.
        public unsafe Src2TgtMaps(MR._ByValue_Src2TgtMaps _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Src2TgtMaps._Underlying *__MR_Src2TgtMaps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Src2TgtMaps._Underlying *_other);
            _UnderlyingPtr = __MR_Src2TgtMaps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Src2TgtMaps::Src2TgtMaps`.
        public unsafe Src2TgtMaps(MR.FaceMap? outFmap, MR.VertMap? outVmap, MR.WholeEdgeMap? outEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Src2TgtMaps_Construct", ExactSpelling = true)]
            extern static MR.Src2TgtMaps._Underlying *__MR_Src2TgtMaps_Construct(MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            _UnderlyingPtr = __MR_Src2TgtMaps_Construct(outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Src2TgtMaps` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Src2TgtMaps`/`Const_Src2TgtMaps` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Src2TgtMaps
    {
        internal readonly Const_Src2TgtMaps? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Src2TgtMaps(Const_Src2TgtMaps new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Src2TgtMaps(Const_Src2TgtMaps arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Src2TgtMaps` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Src2TgtMaps`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Src2TgtMaps`/`Const_Src2TgtMaps` directly.
    public class _InOptMut_Src2TgtMaps
    {
        public Src2TgtMaps? Opt;

        public _InOptMut_Src2TgtMaps() {}
        public _InOptMut_Src2TgtMaps(Src2TgtMaps value) {Opt = value;}
        public static implicit operator _InOptMut_Src2TgtMaps(Src2TgtMaps value) {return new(value);}
    }

    /// This is used for optional parameters of class `Src2TgtMaps` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Src2TgtMaps`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Src2TgtMaps`/`Const_Src2TgtMaps` to pass it to the function.
    public class _InOptConst_Src2TgtMaps
    {
        public Const_Src2TgtMaps? Opt;

        public _InOptConst_Src2TgtMaps() {}
        public _InOptConst_Src2TgtMaps(Const_Src2TgtMaps value) {Opt = value;}
        public static implicit operator _InOptConst_Src2TgtMaps(Const_Src2TgtMaps value) {return new(value);}
    }

    /// use this adapter to call functions expecting PartMapping parameter to receive tgt2src dense maps
    /// Generated from class `MR::Tgt2SrcMaps`.
    /// This is the const half of the class.
    public class Const_Tgt2SrcMaps : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Tgt2SrcMaps(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_Destroy", ExactSpelling = true)]
            extern static void __MR_Tgt2SrcMaps_Destroy(_Underlying *_this);
            __MR_Tgt2SrcMaps_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Tgt2SrcMaps() {Dispose(false);}

        /// Generated from constructor `MR::Tgt2SrcMaps::Tgt2SrcMaps`.
        public unsafe Const_Tgt2SrcMaps(MR._ByValue_Tgt2SrcMaps _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Tgt2SrcMaps._Underlying *__MR_Tgt2SrcMaps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Tgt2SrcMaps._Underlying *_other);
            _UnderlyingPtr = __MR_Tgt2SrcMaps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Tgt2SrcMaps::Tgt2SrcMaps`.
        public unsafe Const_Tgt2SrcMaps(MR.FaceMap? outFmap, MR.VertMap? outVmap, MR.WholeEdgeMap? outEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_Construct", ExactSpelling = true)]
            extern static MR.Tgt2SrcMaps._Underlying *__MR_Tgt2SrcMaps_Construct(MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            _UnderlyingPtr = __MR_Tgt2SrcMaps_Construct(outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// Generated from conversion operator `MR::Tgt2SrcMaps::operator const MR::PartMapping &`.
        public static unsafe implicit operator MR.Const_PartMapping(MR.Const_Tgt2SrcMaps _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_ConvertTo_const_MR_PartMapping_ref", ExactSpelling = true)]
            extern static MR.Const_PartMapping._Underlying *__MR_Tgt2SrcMaps_ConvertTo_const_MR_PartMapping_ref(MR.Const_Tgt2SrcMaps._Underlying *_this);
            return new(__MR_Tgt2SrcMaps_ConvertTo_const_MR_PartMapping_ref(_this._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Tgt2SrcMaps::getPartMapping`.
        public unsafe MR.Const_PartMapping GetPartMapping()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_getPartMapping", ExactSpelling = true)]
            extern static MR.Const_PartMapping._Underlying *__MR_Tgt2SrcMaps_getPartMapping(_Underlying *_this);
            return new(__MR_Tgt2SrcMaps_getPartMapping(_UnderlyingPtr), is_owning: false);
        }
    }

    /// use this adapter to call functions expecting PartMapping parameter to receive tgt2src dense maps
    /// Generated from class `MR::Tgt2SrcMaps`.
    /// This is the non-const half of the class.
    public class Tgt2SrcMaps : Const_Tgt2SrcMaps
    {
        internal unsafe Tgt2SrcMaps(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::Tgt2SrcMaps::Tgt2SrcMaps`.
        public unsafe Tgt2SrcMaps(MR._ByValue_Tgt2SrcMaps _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Tgt2SrcMaps._Underlying *__MR_Tgt2SrcMaps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Tgt2SrcMaps._Underlying *_other);
            _UnderlyingPtr = __MR_Tgt2SrcMaps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Tgt2SrcMaps::Tgt2SrcMaps`.
        public unsafe Tgt2SrcMaps(MR.FaceMap? outFmap, MR.VertMap? outVmap, MR.WholeEdgeMap? outEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Tgt2SrcMaps_Construct", ExactSpelling = true)]
            extern static MR.Tgt2SrcMaps._Underlying *__MR_Tgt2SrcMaps_Construct(MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            _UnderlyingPtr = __MR_Tgt2SrcMaps_Construct(outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Tgt2SrcMaps` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Tgt2SrcMaps`/`Const_Tgt2SrcMaps` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Tgt2SrcMaps
    {
        internal readonly Const_Tgt2SrcMaps? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Tgt2SrcMaps(Const_Tgt2SrcMaps new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Tgt2SrcMaps(Const_Tgt2SrcMaps arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Tgt2SrcMaps` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Tgt2SrcMaps`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Tgt2SrcMaps`/`Const_Tgt2SrcMaps` directly.
    public class _InOptMut_Tgt2SrcMaps
    {
        public Tgt2SrcMaps? Opt;

        public _InOptMut_Tgt2SrcMaps() {}
        public _InOptMut_Tgt2SrcMaps(Tgt2SrcMaps value) {Opt = value;}
        public static implicit operator _InOptMut_Tgt2SrcMaps(Tgt2SrcMaps value) {return new(value);}
    }

    /// This is used for optional parameters of class `Tgt2SrcMaps` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Tgt2SrcMaps`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Tgt2SrcMaps`/`Const_Tgt2SrcMaps` to pass it to the function.
    public class _InOptConst_Tgt2SrcMaps
    {
        public Const_Tgt2SrcMaps? Opt;

        public _InOptConst_Tgt2SrcMaps() {}
        public _InOptConst_Tgt2SrcMaps(Const_Tgt2SrcMaps value) {Opt = value;}
        public static implicit operator _InOptConst_Tgt2SrcMaps(Const_Tgt2SrcMaps value) {return new(value);}
    }
}

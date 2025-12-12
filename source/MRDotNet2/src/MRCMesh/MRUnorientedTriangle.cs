public static partial class MR
{
    /// describes a triangle as three vertex IDs sorted in a way to quickly find same triangles
    /// irrespective of vertex order (clockwise or counterclockwise)
    /// Generated from class `MR::UnorientedTriangle`.
    /// This is the const half of the class.
    public class Const_UnorientedTriangle : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_UnorientedTriangle>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UnorientedTriangle(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_Destroy", ExactSpelling = true)]
            extern static void __MR_UnorientedTriangle_Destroy(_Underlying *_this);
            __MR_UnorientedTriangle_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UnorientedTriangle() {Dispose(false);}

        public unsafe MR.Std.Const_Array_MRVertId_3 Verts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_Get_verts", ExactSpelling = true)]
                extern static MR.Std.Const_Array_MRVertId_3._Underlying *__MR_UnorientedTriangle_Get_verts(_Underlying *_this);
                return new(__MR_UnorientedTriangle_Get_verts(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::UnorientedTriangle::UnorientedTriangle`.
        public unsafe Const_UnorientedTriangle(MR.Const_UnorientedTriangle _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnorientedTriangle._Underlying *__MR_UnorientedTriangle_ConstructFromAnother(MR.UnorientedTriangle._Underlying *_other);
            _UnderlyingPtr = __MR_UnorientedTriangle_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::UnorientedTriangle::UnorientedTriangle`.
        public unsafe Const_UnorientedTriangle(MR.Std.Const_Array_MRVertId_3 inVs, MR.Misc.InOut<bool>? outFlipped = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_Construct", ExactSpelling = true)]
            extern static MR.UnorientedTriangle._Underlying *__MR_UnorientedTriangle_Construct(MR.Std.Const_Array_MRVertId_3._Underlying *inVs, bool *outFlipped);
            bool __value_outFlipped = outFlipped is not null ? outFlipped.Value : default(bool);
            _UnderlyingPtr = __MR_UnorientedTriangle_Construct(inVs._UnderlyingPtr, outFlipped is not null ? &__value_outFlipped : null);
            if (outFlipped is not null) outFlipped.Value = __value_outFlipped;
        }

        /// Generated from conversion operator `MR::UnorientedTriangle::operator const std::array<MR::VertId, 3> &`.
        public static unsafe implicit operator MR.Std.Const_Array_MRVertId_3(MR.Const_UnorientedTriangle _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_ConvertTo_const_std_array_MR_VertId_3_ref", ExactSpelling = true)]
            extern static MR.Std.Const_Array_MRVertId_3._Underlying *__MR_UnorientedTriangle_ConvertTo_const_std_array_MR_VertId_3_ref(MR.Const_UnorientedTriangle._Underlying *_this);
            return new(__MR_UnorientedTriangle_ConvertTo_const_std_array_MR_VertId_3_ref(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns this triangle with the opposite orientation
        /// Generated from method `MR::UnorientedTriangle::getFlipped`.
        public unsafe MR.Std.Array_MRVertId_3 GetFlipped()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_getFlipped", ExactSpelling = true)]
            extern static MR.Std.Array_MRVertId_3 __MR_UnorientedTriangle_getFlipped(_Underlying *_this);
            return __MR_UnorientedTriangle_getFlipped(_UnderlyingPtr);
        }

        /// Generated from method `MR::UnorientedTriangle::operator[]`.
        public unsafe MR.Const_VertId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_index_const", ExactSpelling = true)]
            extern static MR.Const_VertId._Underlying *__MR_UnorientedTriangle_index_const(_Underlying *_this, ulong i);
            return new(__MR_UnorientedTriangle_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_UnorientedTriangle a, MR.Const_UnorientedTriangle b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_UnorientedTriangle", ExactSpelling = true)]
            extern static byte __MR_equal_MR_UnorientedTriangle(MR.Const_UnorientedTriangle._Underlying *a, MR.Const_UnorientedTriangle._Underlying *b);
            return __MR_equal_MR_UnorientedTriangle(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_UnorientedTriangle a, MR.Const_UnorientedTriangle b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_UnorientedTriangle? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_UnorientedTriangle)
                return this == (MR.Const_UnorientedTriangle)other;
            return false;
        }
    }

    /// describes a triangle as three vertex IDs sorted in a way to quickly find same triangles
    /// irrespective of vertex order (clockwise or counterclockwise)
    /// Generated from class `MR::UnorientedTriangle`.
    /// This is the non-const half of the class.
    public class UnorientedTriangle : Const_UnorientedTriangle
    {
        internal unsafe UnorientedTriangle(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Mut_Array_MRVertId_3 Verts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_GetMutable_verts", ExactSpelling = true)]
                extern static MR.Std.Mut_Array_MRVertId_3._Underlying *__MR_UnorientedTriangle_GetMutable_verts(_Underlying *_this);
                return new(__MR_UnorientedTriangle_GetMutable_verts(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::UnorientedTriangle::UnorientedTriangle`.
        public unsafe UnorientedTriangle(MR.Const_UnorientedTriangle _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnorientedTriangle._Underlying *__MR_UnorientedTriangle_ConstructFromAnother(MR.UnorientedTriangle._Underlying *_other);
            _UnderlyingPtr = __MR_UnorientedTriangle_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::UnorientedTriangle::UnorientedTriangle`.
        public unsafe UnorientedTriangle(MR.Std.Const_Array_MRVertId_3 inVs, MR.Misc.InOut<bool>? outFlipped = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_Construct", ExactSpelling = true)]
            extern static MR.UnorientedTriangle._Underlying *__MR_UnorientedTriangle_Construct(MR.Std.Const_Array_MRVertId_3._Underlying *inVs, bool *outFlipped);
            bool __value_outFlipped = outFlipped is not null ? outFlipped.Value : default(bool);
            _UnderlyingPtr = __MR_UnorientedTriangle_Construct(inVs._UnderlyingPtr, outFlipped is not null ? &__value_outFlipped : null);
            if (outFlipped is not null) outFlipped.Value = __value_outFlipped;
        }

        /// Generated from conversion operator `MR::UnorientedTriangle::operator std::array<MR::VertId, 3> &`.
        public static unsafe implicit operator MR.Std.Mut_Array_MRVertId_3(MR.UnorientedTriangle _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_ConvertTo_std_array_MR_VertId_3_ref", ExactSpelling = true)]
            extern static MR.Std.Mut_Array_MRVertId_3._Underlying *__MR_UnorientedTriangle_ConvertTo_std_array_MR_VertId_3_ref(MR.UnorientedTriangle._Underlying *_this);
            return new(__MR_UnorientedTriangle_ConvertTo_std_array_MR_VertId_3_ref(_this._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UnorientedTriangle::operator=`.
        public unsafe MR.UnorientedTriangle Assign(MR.Const_UnorientedTriangle _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UnorientedTriangle._Underlying *__MR_UnorientedTriangle_AssignFromAnother(_Underlying *_this, MR.UnorientedTriangle._Underlying *_other);
            return new(__MR_UnorientedTriangle_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UnorientedTriangle::operator[]`.
        public unsafe new MR.Mut_VertId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnorientedTriangle_index", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_UnorientedTriangle_index(_Underlying *_this, ulong i);
            return new(__MR_UnorientedTriangle_index(_UnderlyingPtr, i), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `UnorientedTriangle` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UnorientedTriangle`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnorientedTriangle`/`Const_UnorientedTriangle` directly.
    public class _InOptMut_UnorientedTriangle
    {
        public UnorientedTriangle? Opt;

        public _InOptMut_UnorientedTriangle() {}
        public _InOptMut_UnorientedTriangle(UnorientedTriangle value) {Opt = value;}
        public static implicit operator _InOptMut_UnorientedTriangle(UnorientedTriangle value) {return new(value);}
    }

    /// This is used for optional parameters of class `UnorientedTriangle` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UnorientedTriangle`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnorientedTriangle`/`Const_UnorientedTriangle` to pass it to the function.
    public class _InOptConst_UnorientedTriangle
    {
        public Const_UnorientedTriangle? Opt;

        public _InOptConst_UnorientedTriangle() {}
        public _InOptConst_UnorientedTriangle(Const_UnorientedTriangle value) {Opt = value;}
        public static implicit operator _InOptConst_UnorientedTriangle(Const_UnorientedTriangle value) {return new(value);}
    }
}

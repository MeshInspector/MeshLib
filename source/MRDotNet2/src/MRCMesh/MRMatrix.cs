public static partial class MR
{
    /// Row-major matrix with T values
    /// Generated from class `MR::Matrix<float>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RectIndexer`
    /// This is the const half of the class.
    public class Const_Matrix_Float : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_Matrix_float_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_Matrix_float_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_Matrix_float_UseCount();
                return __MR_std_shared_ptr_MR_Matrix_float_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Matrix_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_Matrix_float_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_Matrix_Float(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Matrix_float_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Matrix_float_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Matrix_float_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Matrix_float_ConstructNonOwning(ptr);
        }

        internal unsafe Const_Matrix_Float(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe Matrix_Float _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Matrix_float_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_Matrix_float_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Matrix_float_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Matrix_float_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Matrix_float_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_Matrix_float_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_Matrix_float_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix_Float() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RectIndexer(Const_Matrix_Float self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_UpcastTo_MR_RectIndexer", ExactSpelling = true)]
            extern static MR.Const_RectIndexer._Underlying *__MR_Matrix_float_UpcastTo_MR_RectIndexer(_Underlying *_this);
            return MR.Const_RectIndexer._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_Matrix_float_UpcastTo_MR_RectIndexer(self._UnderlyingPtr));
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix_Float() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_DefaultConstruct();
            _LateMakeShared(__MR_Matrix_float_DefaultConstruct());
        }

        /// Generated from constructor `MR::Matrix<float>::Matrix`.
        public unsafe Const_Matrix_Float(MR._ByValue_Matrix_Float _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._Underlying *_other);
            _LateMakeShared(__MR_Matrix_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::Matrix<float>::Matrix`.
        public unsafe Const_Matrix_Float(ulong numRows, ulong numCols) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_Construct", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_Construct(ulong numRows, ulong numCols);
            _LateMakeShared(__MR_Matrix_float_Construct(numRows, numCols));
        }

        /// Generated from method `MR::Matrix<float>::operator()`.
        public unsafe float Call(ulong row, ulong col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_call_const_2", ExactSpelling = true)]
            extern static float *__MR_Matrix_float_call_const_2(_Underlying *_this, ulong row, ulong col);
            return *__MR_Matrix_float_call_const_2(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix<float>::operator()`.
        public unsafe float Call(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_call_const_1", ExactSpelling = true)]
            extern static float *__MR_Matrix_float_call_const_1(_Underlying *_this, ulong i);
            return *__MR_Matrix_float_call_const_1(_UnderlyingPtr, i);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix<float>::transposed`.
        public unsafe MR.Misc._Moved<MR.Matrix_Float> Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_transposed", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_transposed(_Underlying *_this);
            return MR.Misc.Move(new MR.Matrix_Float(__MR_Matrix_float_transposed(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Matrix<float>::getRowsNum`.
        public unsafe ulong GetRowsNum()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_getRowsNum", ExactSpelling = true)]
            extern static ulong __MR_Matrix_float_getRowsNum(_Underlying *_this);
            return __MR_Matrix_float_getRowsNum(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix<float>::getColsNum`.
        public unsafe ulong GetColsNum()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_getColsNum", ExactSpelling = true)]
            extern static ulong __MR_Matrix_float_getColsNum(_Underlying *_this);
            return __MR_Matrix_float_getColsNum(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix<float>::data`.
        public unsafe MR.Std.Const_Vector_Float Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_data", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_Float._Underlying *__MR_Matrix_float_data(_Underlying *_this);
            return new(__MR_Matrix_float_data(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Matrix<float>::dims`.
        public unsafe MR.Const_Vector2i Dims()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_dims", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_Matrix_float_dims(_Underlying *_this);
            return new(__MR_Matrix_float_dims(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Matrix<float>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_size", ExactSpelling = true)]
            extern static ulong __MR_Matrix_float_size(_Underlying *_this);
            return __MR_Matrix_float_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix<float>::toPixelId`.
        public unsafe MR.PixelId ToPixelId(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_toPixelId", ExactSpelling = true)]
            extern static MR.PixelId __MR_Matrix_float_toPixelId(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_Matrix_float_toPixelId(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix<float>::toIndex`.
        public unsafe ulong ToIndex(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_toIndex", ExactSpelling = true)]
            extern static ulong __MR_Matrix_float_toIndex(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_Matrix_float_toIndex(_UnderlyingPtr, pos._UnderlyingPtr);
        }
    }

    /// Row-major matrix with T values
    /// Generated from class `MR::Matrix<float>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RectIndexer`
    /// This is the non-const half of the class.
    public class Matrix_Float : Const_Matrix_Float
    {
        internal unsafe Matrix_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe Matrix_Float(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RectIndexer(Matrix_Float self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_UpcastTo_MR_RectIndexer", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_Matrix_float_UpcastTo_MR_RectIndexer(_Underlying *_this);
            return MR.RectIndexer._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_Matrix_float_UpcastTo_MR_RectIndexer(self._UnderlyingPtr));
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix_Float() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_DefaultConstruct();
            _LateMakeShared(__MR_Matrix_float_DefaultConstruct());
        }

        /// Generated from constructor `MR::Matrix<float>::Matrix`.
        public unsafe Matrix_Float(MR._ByValue_Matrix_Float _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._Underlying *_other);
            _LateMakeShared(__MR_Matrix_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::Matrix<float>::Matrix`.
        public unsafe Matrix_Float(ulong numRows, ulong numCols) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_Construct", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_Construct(ulong numRows, ulong numCols);
            _LateMakeShared(__MR_Matrix_float_Construct(numRows, numCols));
        }

        /// Generated from method `MR::Matrix<float>::operator=`.
        public unsafe MR.Matrix_Float Assign(MR._ByValue_Matrix_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._Underlying *_other);
            return new(__MR_Matrix_float_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// main access method
        /// Generated from method `MR::Matrix<float>::operator()`.
        public unsafe new ref float Call(ulong row, ulong col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_call_2", ExactSpelling = true)]
            extern static float *__MR_Matrix_float_call_2(_Underlying *_this, ulong row, ulong col);
            return ref *__MR_Matrix_float_call_2(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix<float>::operator()`.
        public unsafe new ref float Call(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_call_1", ExactSpelling = true)]
            extern static float *__MR_Matrix_float_call_1(_Underlying *_this, ulong i);
            return ref *__MR_Matrix_float_call_1(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::Matrix<float>::getSubMatrix`.
        public unsafe MR.Misc._Moved<MR.Matrix_Float> GetSubMatrix(ulong startRow, ulong nRow, ulong startCol, ulong nCol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_getSubMatrix", ExactSpelling = true)]
            extern static MR.Matrix_Float._Underlying *__MR_Matrix_float_getSubMatrix(_Underlying *_this, ulong startRow, ulong nRow, ulong startCol, ulong nCol);
            return MR.Misc.Move(new MR.Matrix_Float(__MR_Matrix_float_getSubMatrix(_UnderlyingPtr, startRow, nRow, startCol, nCol), is_owning: true));
        }

        /// Generated from method `MR::Matrix<float>::fill`.
        public unsafe void Fill(float val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_fill", ExactSpelling = true)]
            extern static void __MR_Matrix_float_fill(_Underlying *_this, float val);
            __MR_Matrix_float_fill(_UnderlyingPtr, val);
        }

        /// Generated from method `MR::Matrix<float>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_clear", ExactSpelling = true)]
            extern static void __MR_Matrix_float_clear(_Underlying *_this);
            __MR_Matrix_float_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix<float>::resize`.
        public unsafe void Resize(MR.Const_Vector2i dims)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix_float_resize", ExactSpelling = true)]
            extern static void __MR_Matrix_float_resize(_Underlying *_this, MR.Const_Vector2i._Underlying *dims);
            __MR_Matrix_float_resize(_UnderlyingPtr, dims._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Matrix_Float` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Matrix_Float`/`Const_Matrix_Float` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Matrix_Float
    {
        internal readonly Const_Matrix_Float? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Matrix_Float() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Matrix_Float(Const_Matrix_Float new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Matrix_Float(Const_Matrix_Float arg) {return new(arg);}
        public _ByValue_Matrix_Float(MR.Misc._Moved<Matrix_Float> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Matrix_Float(MR.Misc._Moved<Matrix_Float> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Matrix_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix_Float`/`Const_Matrix_Float` directly.
    public class _InOptMut_Matrix_Float
    {
        public Matrix_Float? Opt;

        public _InOptMut_Matrix_Float() {}
        public _InOptMut_Matrix_Float(Matrix_Float value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix_Float(Matrix_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `Matrix_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix_Float`/`Const_Matrix_Float` to pass it to the function.
    public class _InOptConst_Matrix_Float
    {
        public Const_Matrix_Float? Opt;

        public _InOptConst_Matrix_Float() {}
        public _InOptConst_Matrix_Float(Const_Matrix_Float value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix_Float(Const_Matrix_Float value) {return new(value);}
    }
}

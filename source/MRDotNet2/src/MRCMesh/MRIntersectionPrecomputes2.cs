public static partial class MR
{
    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes2<float>`.
    /// This is the const half of the class.
    public class Const_IntersectionPrecomputes2_Float : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IntersectionPrecomputes2_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Destroy", ExactSpelling = true)]
            extern static void __MR_IntersectionPrecomputes2_float_Destroy(_Underlying *_this);
            __MR_IntersectionPrecomputes2_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IntersectionPrecomputes2_Float() {Dispose(false);}

        // {1 / dir}
        public unsafe MR.Const_Vector2f InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_invDir", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_IntersectionPrecomputes2_float_Get_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_float_Get_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2} => {1,0}
        public unsafe int MaxDimIdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_maxDimIdxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_float_Get_maxDimIdxY(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_float_Get_maxDimIdxY(_UnderlyingPtr);
            }
        }

        public unsafe int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_float_Get_idxX(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_float_Get_idxX(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public unsafe MR.Const_Vector2i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_sign", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_IntersectionPrecomputes2_float_Get_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_float_Get_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public unsafe float Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_Sx", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes2_float_Get_Sx(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_float_Get_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe float Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Get_Sy", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes2_float_Get_Sy(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_float_Get_Sy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IntersectionPrecomputes2_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public unsafe Const_IntersectionPrecomputes2_Float(MR.Const_IntersectionPrecomputes2_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_ConstructFromAnother(MR.IntersectionPrecomputes2_Float._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public unsafe Const_IntersectionPrecomputes2_Float(MR.Const_Vector2f dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_Construct(MR.Const_Vector2f._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public static unsafe implicit operator Const_IntersectionPrecomputes2_Float(MR.Const_Vector2f dir) {return new(dir);}
    }

    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes2<float>`.
    /// This is the non-const half of the class.
    public class IntersectionPrecomputes2_Float : Const_IntersectionPrecomputes2_Float
    {
        internal unsafe IntersectionPrecomputes2_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // {1 / dir}
        public new unsafe MR.Mut_Vector2f InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_invDir", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_IntersectionPrecomputes2_float_GetMutable_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_float_GetMutable_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2} => {1,0}
        public new unsafe ref int MaxDimIdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_maxDimIdxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_float_GetMutable_maxDimIdxY(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_float_GetMutable_maxDimIdxY(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_float_GetMutable_idxX(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_float_GetMutable_idxX(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public new unsafe MR.Mut_Vector2i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_sign", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_IntersectionPrecomputes2_float_GetMutable_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_float_GetMutable_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public new unsafe ref float Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_Sx", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes2_float_GetMutable_Sx(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_float_GetMutable_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref float Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_GetMutable_Sy", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes2_float_GetMutable_Sy(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_float_GetMutable_Sy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IntersectionPrecomputes2_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public unsafe IntersectionPrecomputes2_Float(MR.Const_IntersectionPrecomputes2_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_ConstructFromAnother(MR.IntersectionPrecomputes2_Float._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public unsafe IntersectionPrecomputes2_Float(MR.Const_Vector2f dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_Construct(MR.Const_Vector2f._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_float_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public static unsafe implicit operator IntersectionPrecomputes2_Float(MR.Const_Vector2f dir) {return new(dir);}

        /// Generated from method `MR::IntersectionPrecomputes2<float>::operator=`.
        public unsafe MR.IntersectionPrecomputes2_Float Assign(MR.Const_IntersectionPrecomputes2_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Float._Underlying *__MR_IntersectionPrecomputes2_float_AssignFromAnother(_Underlying *_this, MR.IntersectionPrecomputes2_Float._Underlying *_other);
            return new(__MR_IntersectionPrecomputes2_float_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes2_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionPrecomputes2_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes2_Float`/`Const_IntersectionPrecomputes2_Float` directly.
    public class _InOptMut_IntersectionPrecomputes2_Float
    {
        public IntersectionPrecomputes2_Float? Opt;

        public _InOptMut_IntersectionPrecomputes2_Float() {}
        public _InOptMut_IntersectionPrecomputes2_Float(IntersectionPrecomputes2_Float value) {Opt = value;}
        public static implicit operator _InOptMut_IntersectionPrecomputes2_Float(IntersectionPrecomputes2_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes2_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionPrecomputes2_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes2_Float`/`Const_IntersectionPrecomputes2_Float` to pass it to the function.
    public class _InOptConst_IntersectionPrecomputes2_Float
    {
        public Const_IntersectionPrecomputes2_Float? Opt;

        public _InOptConst_IntersectionPrecomputes2_Float() {}
        public _InOptConst_IntersectionPrecomputes2_Float(Const_IntersectionPrecomputes2_Float value) {Opt = value;}
        public static implicit operator _InOptConst_IntersectionPrecomputes2_Float(Const_IntersectionPrecomputes2_Float value) {return new(value);}

        /// Generated from constructor `MR::IntersectionPrecomputes2<float>::IntersectionPrecomputes2`.
        public static unsafe implicit operator _InOptConst_IntersectionPrecomputes2_Float(MR.Const_Vector2f dir) {return new MR.IntersectionPrecomputes2_Float(dir);}
    }

    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes2<double>`.
    /// This is the const half of the class.
    public class Const_IntersectionPrecomputes2_Double : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IntersectionPrecomputes2_Double(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Destroy", ExactSpelling = true)]
            extern static void __MR_IntersectionPrecomputes2_double_Destroy(_Underlying *_this);
            __MR_IntersectionPrecomputes2_double_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IntersectionPrecomputes2_Double() {Dispose(false);}

        // {1 / dir}
        public unsafe MR.Const_Vector2d InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_invDir", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_IntersectionPrecomputes2_double_Get_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_double_Get_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2} => {1,0}
        public unsafe int MaxDimIdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_maxDimIdxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_double_Get_maxDimIdxY(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_double_Get_maxDimIdxY(_UnderlyingPtr);
            }
        }

        public unsafe int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_double_Get_idxX(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_double_Get_idxX(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public unsafe MR.Const_Vector2i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_sign", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_IntersectionPrecomputes2_double_Get_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_double_Get_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public unsafe double Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_Sx", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes2_double_Get_Sx(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_double_Get_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe double Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Get_Sy", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes2_double_Get_Sy(_Underlying *_this);
                return *__MR_IntersectionPrecomputes2_double_Get_Sy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IntersectionPrecomputes2_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public unsafe Const_IntersectionPrecomputes2_Double(MR.Const_IntersectionPrecomputes2_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_ConstructFromAnother(MR.IntersectionPrecomputes2_Double._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public unsafe Const_IntersectionPrecomputes2_Double(MR.Const_Vector2d dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_Construct(MR.Const_Vector2d._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public static unsafe implicit operator Const_IntersectionPrecomputes2_Double(MR.Const_Vector2d dir) {return new(dir);}
    }

    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes2<double>`.
    /// This is the non-const half of the class.
    public class IntersectionPrecomputes2_Double : Const_IntersectionPrecomputes2_Double
    {
        internal unsafe IntersectionPrecomputes2_Double(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // {1 / dir}
        public new unsafe MR.Mut_Vector2d InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_invDir", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_IntersectionPrecomputes2_double_GetMutable_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_double_GetMutable_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2} => {1,0}
        public new unsafe ref int MaxDimIdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_maxDimIdxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_double_GetMutable_maxDimIdxY(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_double_GetMutable_maxDimIdxY(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes2_double_GetMutable_idxX(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_double_GetMutable_idxX(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public new unsafe MR.Mut_Vector2i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_sign", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_IntersectionPrecomputes2_double_GetMutable_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes2_double_GetMutable_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public new unsafe ref double Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_Sx", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes2_double_GetMutable_Sx(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_double_GetMutable_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref double Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_GetMutable_Sy", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes2_double_GetMutable_Sy(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes2_double_GetMutable_Sy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IntersectionPrecomputes2_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public unsafe IntersectionPrecomputes2_Double(MR.Const_IntersectionPrecomputes2_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_ConstructFromAnother(MR.IntersectionPrecomputes2_Double._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public unsafe IntersectionPrecomputes2_Double(MR.Const_Vector2d dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_Construct(MR.Const_Vector2d._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes2_double_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public static unsafe implicit operator IntersectionPrecomputes2_Double(MR.Const_Vector2d dir) {return new(dir);}

        /// Generated from method `MR::IntersectionPrecomputes2<double>::operator=`.
        public unsafe MR.IntersectionPrecomputes2_Double Assign(MR.Const_IntersectionPrecomputes2_Double _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes2_double_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes2_Double._Underlying *__MR_IntersectionPrecomputes2_double_AssignFromAnother(_Underlying *_this, MR.IntersectionPrecomputes2_Double._Underlying *_other);
            return new(__MR_IntersectionPrecomputes2_double_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes2_Double` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionPrecomputes2_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes2_Double`/`Const_IntersectionPrecomputes2_Double` directly.
    public class _InOptMut_IntersectionPrecomputes2_Double
    {
        public IntersectionPrecomputes2_Double? Opt;

        public _InOptMut_IntersectionPrecomputes2_Double() {}
        public _InOptMut_IntersectionPrecomputes2_Double(IntersectionPrecomputes2_Double value) {Opt = value;}
        public static implicit operator _InOptMut_IntersectionPrecomputes2_Double(IntersectionPrecomputes2_Double value) {return new(value);}
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes2_Double` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionPrecomputes2_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes2_Double`/`Const_IntersectionPrecomputes2_Double` to pass it to the function.
    public class _InOptConst_IntersectionPrecomputes2_Double
    {
        public Const_IntersectionPrecomputes2_Double? Opt;

        public _InOptConst_IntersectionPrecomputes2_Double() {}
        public _InOptConst_IntersectionPrecomputes2_Double(Const_IntersectionPrecomputes2_Double value) {Opt = value;}
        public static implicit operator _InOptConst_IntersectionPrecomputes2_Double(Const_IntersectionPrecomputes2_Double value) {return new(value);}

        /// Generated from constructor `MR::IntersectionPrecomputes2<double>::IntersectionPrecomputes2`.
        public static unsafe implicit operator _InOptConst_IntersectionPrecomputes2_Double(MR.Const_Vector2d dir) {return new MR.IntersectionPrecomputes2_Double(dir);}
    }
}

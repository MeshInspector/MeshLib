public static partial class MR
{
    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes<double>`.
    /// This is the const half of the class.
    public class Const_IntersectionPrecomputes_Double : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IntersectionPrecomputes_Double(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Destroy", ExactSpelling = true)]
            extern static void __MR_IntersectionPrecomputes_double_Destroy(_Underlying *_this);
            __MR_IntersectionPrecomputes_double_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IntersectionPrecomputes_Double() {Dispose(false);}

        // {1 / dir}
        public unsafe MR.Const_Vector3d InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_invDir", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_IntersectionPrecomputes_double_Get_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes_double_Get_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2,-3} => {2,1,0}
        public unsafe int MaxDimIdxZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_maxDimIdxZ", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_Get_maxDimIdxZ(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_maxDimIdxZ(_UnderlyingPtr);
            }
        }

        public unsafe int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_Get_idxX(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_idxX(_UnderlyingPtr);
            }
        }

        public unsafe int IdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_idxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_Get_idxY(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_idxY(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public unsafe MR.Const_Vector3i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_sign", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_IntersectionPrecomputes_double_Get_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes_double_Get_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public unsafe double Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_Sx", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_Get_Sx(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe double Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_Sy", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_Get_Sy(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_Sy(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe double Sz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Get_Sz", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_Get_Sz(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_double_Get_Sz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IntersectionPrecomputes_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public unsafe Const_IntersectionPrecomputes_Double(MR.Const_IntersectionPrecomputes_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_ConstructFromAnother(MR.IntersectionPrecomputes_Double._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public unsafe Const_IntersectionPrecomputes_Double(MR.Const_Vector3d dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_Construct(MR.Const_Vector3d._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public static unsafe implicit operator Const_IntersectionPrecomputes_Double(MR.Const_Vector3d dir) {return new(dir);}
    }

    /// stores useful precomputed values for presented direction vector
    /// \details allows to avoid repeatable computations during intersection finding
    /// Generated from class `MR::IntersectionPrecomputes<double>`.
    /// This is the non-const half of the class.
    public class IntersectionPrecomputes_Double : Const_IntersectionPrecomputes_Double
    {
        internal unsafe IntersectionPrecomputes_Double(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // {1 / dir}
        public new unsafe MR.Mut_Vector3d InvDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_invDir", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_IntersectionPrecomputes_double_GetMutable_invDir(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes_double_GetMutable_invDir(_UnderlyingPtr), is_owning: false);
            }
        }

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2,-3} => {2,1,0}
        public new unsafe ref int MaxDimIdxZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_maxDimIdxZ", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_GetMutable_maxDimIdxZ(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_maxDimIdxZ(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_GetMutable_idxX(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_idxX(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_idxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_double_GetMutable_idxY(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_idxY(_UnderlyingPtr);
            }
        }

        /// stores signs of direction vector;
        public new unsafe MR.Mut_Vector3i Sign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_sign", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_IntersectionPrecomputes_double_GetMutable_sign(_Underlying *_this);
                return new(__MR_IntersectionPrecomputes_double_GetMutable_sign(_UnderlyingPtr), is_owning: false);
            }
        }

        /// precomputed factors
        public new unsafe ref double Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_Sx", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_GetMutable_Sx(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref double Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_Sy", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_GetMutable_Sy(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_Sy(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref double Sz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_GetMutable_Sz", ExactSpelling = true)]
                extern static double *__MR_IntersectionPrecomputes_double_GetMutable_Sz(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_double_GetMutable_Sz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IntersectionPrecomputes_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public unsafe IntersectionPrecomputes_Double(MR.Const_IntersectionPrecomputes_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_ConstructFromAnother(MR.IntersectionPrecomputes_Double._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public unsafe IntersectionPrecomputes_Double(MR.Const_Vector3d dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_Construct(MR.Const_Vector3d._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_double_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public static unsafe implicit operator IntersectionPrecomputes_Double(MR.Const_Vector3d dir) {return new(dir);}

        /// Generated from method `MR::IntersectionPrecomputes<double>::operator=`.
        public unsafe MR.IntersectionPrecomputes_Double Assign(MR.Const_IntersectionPrecomputes_Double _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_double_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Double._Underlying *__MR_IntersectionPrecomputes_double_AssignFromAnother(_Underlying *_this, MR.IntersectionPrecomputes_Double._Underlying *_other);
            return new(__MR_IntersectionPrecomputes_double_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes_Double` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionPrecomputes_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes_Double`/`Const_IntersectionPrecomputes_Double` directly.
    public class _InOptMut_IntersectionPrecomputes_Double
    {
        public IntersectionPrecomputes_Double? Opt;

        public _InOptMut_IntersectionPrecomputes_Double() {}
        public _InOptMut_IntersectionPrecomputes_Double(IntersectionPrecomputes_Double value) {Opt = value;}
        public static implicit operator _InOptMut_IntersectionPrecomputes_Double(IntersectionPrecomputes_Double value) {return new(value);}
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes_Double` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionPrecomputes_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes_Double`/`Const_IntersectionPrecomputes_Double` to pass it to the function.
    public class _InOptConst_IntersectionPrecomputes_Double
    {
        public Const_IntersectionPrecomputes_Double? Opt;

        public _InOptConst_IntersectionPrecomputes_Double() {}
        public _InOptConst_IntersectionPrecomputes_Double(Const_IntersectionPrecomputes_Double value) {Opt = value;}
        public static implicit operator _InOptConst_IntersectionPrecomputes_Double(Const_IntersectionPrecomputes_Double value) {return new(value);}

        /// Generated from constructor `MR::IntersectionPrecomputes<double>::IntersectionPrecomputes`.
        public static unsafe implicit operator _InOptConst_IntersectionPrecomputes_Double(MR.Const_Vector3d dir) {return new MR.IntersectionPrecomputes_Double(dir);}
    }

    /// Generated from class `MR::IntersectionPrecomputes<float>`.
    /// This is the const half of the class.
    public class Const_IntersectionPrecomputes_Float : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IntersectionPrecomputes_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Destroy", ExactSpelling = true)]
            extern static void __MR_IntersectionPrecomputes_float_Destroy(_Underlying *_this);
            __MR_IntersectionPrecomputes_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IntersectionPrecomputes_Float() {Dispose(false);}

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2,-3} => {2,1,0}
        public unsafe int MaxDimIdxZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_maxDimIdxZ", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_Get_maxDimIdxZ(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_maxDimIdxZ(_UnderlyingPtr);
            }
        }

        public unsafe int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_Get_idxX(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_idxX(_UnderlyingPtr);
            }
        }

        public unsafe int IdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_idxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_Get_idxY(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_idxY(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe float Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_Sx", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_Get_Sx(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe float Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_Sy", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_Get_Sy(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_Sy(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public unsafe float Sz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Get_Sz", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_Get_Sz(_Underlying *_this);
                return *__MR_IntersectionPrecomputes_float_Get_Sz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IntersectionPrecomputes_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public unsafe Const_IntersectionPrecomputes_Float(MR.Const_IntersectionPrecomputes_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_ConstructFromAnother(MR.IntersectionPrecomputes_Float._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public unsafe Const_IntersectionPrecomputes_Float(MR.Const_Vector3f dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_Construct(MR.Const_Vector3f._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public static unsafe implicit operator Const_IntersectionPrecomputes_Float(MR.Const_Vector3f dir) {return new(dir);}
    }

    /// Generated from class `MR::IntersectionPrecomputes<float>`.
    /// This is the non-const half of the class.
    public class IntersectionPrecomputes_Float : Const_IntersectionPrecomputes_Float
    {
        internal unsafe IntersectionPrecomputes_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // [0]max, [1]next, [2]next-next
        // f.e. {1,2,-3} => {2,1,0}
        public new unsafe ref int MaxDimIdxZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_maxDimIdxZ", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_GetMutable_maxDimIdxZ(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_maxDimIdxZ(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxX
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_idxX", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_GetMutable_idxX(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_idxX(_UnderlyingPtr);
            }
        }

        public new unsafe ref int IdxY
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_idxY", ExactSpelling = true)]
                extern static int *__MR_IntersectionPrecomputes_float_GetMutable_idxY(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_idxY(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref float Sx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_Sx", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_GetMutable_Sx(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_Sx(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref float Sy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_Sy", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_GetMutable_Sy(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_Sy(_UnderlyingPtr);
            }
        }

        /// precomputed factors
        public new unsafe ref float Sz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_GetMutable_Sz", ExactSpelling = true)]
                extern static float *__MR_IntersectionPrecomputes_float_GetMutable_Sz(_Underlying *_this);
                return ref *__MR_IntersectionPrecomputes_float_GetMutable_Sz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IntersectionPrecomputes_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_DefaultConstruct();
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_DefaultConstruct();
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public unsafe IntersectionPrecomputes_Float(MR.Const_IntersectionPrecomputes_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_ConstructFromAnother(MR.IntersectionPrecomputes_Float._Underlying *_other);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public unsafe IntersectionPrecomputes_Float(MR.Const_Vector3f dir) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_Construct", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_Construct(MR.Const_Vector3f._Underlying *dir);
            _UnderlyingPtr = __MR_IntersectionPrecomputes_float_Construct(dir._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public static unsafe implicit operator IntersectionPrecomputes_Float(MR.Const_Vector3f dir) {return new(dir);}

        /// Generated from method `MR::IntersectionPrecomputes<float>::operator=`.
        public unsafe MR.IntersectionPrecomputes_Float Assign(MR.Const_IntersectionPrecomputes_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntersectionPrecomputes_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IntersectionPrecomputes_Float._Underlying *__MR_IntersectionPrecomputes_float_AssignFromAnother(_Underlying *_this, MR.IntersectionPrecomputes_Float._Underlying *_other);
            return new(__MR_IntersectionPrecomputes_float_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionPrecomputes_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes_Float`/`Const_IntersectionPrecomputes_Float` directly.
    public class _InOptMut_IntersectionPrecomputes_Float
    {
        public IntersectionPrecomputes_Float? Opt;

        public _InOptMut_IntersectionPrecomputes_Float() {}
        public _InOptMut_IntersectionPrecomputes_Float(IntersectionPrecomputes_Float value) {Opt = value;}
        public static implicit operator _InOptMut_IntersectionPrecomputes_Float(IntersectionPrecomputes_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `IntersectionPrecomputes_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionPrecomputes_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntersectionPrecomputes_Float`/`Const_IntersectionPrecomputes_Float` to pass it to the function.
    public class _InOptConst_IntersectionPrecomputes_Float
    {
        public Const_IntersectionPrecomputes_Float? Opt;

        public _InOptConst_IntersectionPrecomputes_Float() {}
        public _InOptConst_IntersectionPrecomputes_Float(Const_IntersectionPrecomputes_Float value) {Opt = value;}
        public static implicit operator _InOptConst_IntersectionPrecomputes_Float(Const_IntersectionPrecomputes_Float value) {return new(value);}

        /// Generated from constructor `MR::IntersectionPrecomputes<float>::IntersectionPrecomputes`.
        public static unsafe implicit operator _InOptConst_IntersectionPrecomputes_Float(MR.Const_Vector3f dir) {return new MR.IntersectionPrecomputes_Float(dir);}
    }

    /**
    * \brief finds index of maximum axis and stores it into dimZ
    * \details http://jcgt.org/published/0002/01/05/paper.pdf
    * Example input: dir = (1,1,-2). Result: dimZ = 2, dimX = 1, dimY = 0.
    * \param[out] dimX are filled by right-hand rule from dimZ
    * \param[out] dimY are filled by right-hand rule from dimZ
    * \param[out] dimZ index of maximum axis
    */
    /// Generated from function `MR::findMaxVectorDim<float>`.
    public static unsafe void FindMaxVectorDim(ref int dimX, ref int dimY, ref int dimZ, MR.Const_Vector3f dir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxVectorDim", ExactSpelling = true)]
        extern static void __MR_findMaxVectorDim(int *dimX, int *dimY, int *dimZ, MR.Const_Vector3f._Underlying *dir);
        fixed (int *__ptr_dimX = &dimX)
        {
            fixed (int *__ptr_dimY = &dimY)
            {
                fixed (int *__ptr_dimZ = &dimZ)
                {
                    __MR_findMaxVectorDim(__ptr_dimX, __ptr_dimY, __ptr_dimZ, dir._UnderlyingPtr);
                }
            }
        }
    }
}

public static partial class MR
{
    public static partial class Parallel
    {
        /// Generated from class `MR::Parallel::CallSimply`.
        /// This is the const half of the class.
        public class Const_CallSimply : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_CallSimply(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_Destroy", ExactSpelling = true)]
                extern static void __MR_Parallel_CallSimply_Destroy(_Underlying *_this);
                __MR_Parallel_CallSimply_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_CallSimply() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_CallSimply() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimply_DefaultConstruct();
                _UnderlyingPtr = __MR_Parallel_CallSimply_DefaultConstruct();
            }

            /// Generated from constructor `MR::Parallel::CallSimply::CallSimply`.
            public unsafe Const_CallSimply(MR.Parallel.Const_CallSimply _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimply_ConstructFromAnother(MR.Parallel.CallSimply._Underlying *_other);
                _UnderlyingPtr = __MR_Parallel_CallSimply_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Parallel::CallSimply`.
        /// This is the non-const half of the class.
        public class CallSimply : Const_CallSimply
        {
            internal unsafe CallSimply(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe CallSimply() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimply_DefaultConstruct();
                _UnderlyingPtr = __MR_Parallel_CallSimply_DefaultConstruct();
            }

            /// Generated from constructor `MR::Parallel::CallSimply::CallSimply`.
            public unsafe CallSimply(MR.Parallel.Const_CallSimply _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimply_ConstructFromAnother(MR.Parallel.CallSimply._Underlying *_other);
                _UnderlyingPtr = __MR_Parallel_CallSimply_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Parallel::CallSimply::operator=`.
            public unsafe MR.Parallel.CallSimply Assign(MR.Parallel.Const_CallSimply _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimply_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimply_AssignFromAnother(_Underlying *_this, MR.Parallel.CallSimply._Underlying *_other);
                return new(__MR_Parallel_CallSimply_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `CallSimply` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_CallSimply`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CallSimply`/`Const_CallSimply` directly.
        public class _InOptMut_CallSimply
        {
            public CallSimply? Opt;

            public _InOptMut_CallSimply() {}
            public _InOptMut_CallSimply(CallSimply value) {Opt = value;}
            public static implicit operator _InOptMut_CallSimply(CallSimply value) {return new(value);}
        }

        /// This is used for optional parameters of class `CallSimply` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_CallSimply`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CallSimply`/`Const_CallSimply` to pass it to the function.
        public class _InOptConst_CallSimply
        {
            public Const_CallSimply? Opt;

            public _InOptConst_CallSimply() {}
            public _InOptConst_CallSimply(Const_CallSimply value) {Opt = value;}
            public static implicit operator _InOptConst_CallSimply(Const_CallSimply value) {return new(value);}
        }

        /// Generated from class `MR::Parallel::CallSimplyMaker`.
        /// This is the const half of the class.
        public class Const_CallSimplyMaker : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_CallSimplyMaker(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_Destroy", ExactSpelling = true)]
                extern static void __MR_Parallel_CallSimplyMaker_Destroy(_Underlying *_this);
                __MR_Parallel_CallSimplyMaker_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_CallSimplyMaker() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_CallSimplyMaker() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Parallel.CallSimplyMaker._Underlying *__MR_Parallel_CallSimplyMaker_DefaultConstruct();
                _UnderlyingPtr = __MR_Parallel_CallSimplyMaker_DefaultConstruct();
            }

            /// Generated from constructor `MR::Parallel::CallSimplyMaker::CallSimplyMaker`.
            public unsafe Const_CallSimplyMaker(MR.Parallel.Const_CallSimplyMaker _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimplyMaker._Underlying *__MR_Parallel_CallSimplyMaker_ConstructFromAnother(MR.Parallel.CallSimplyMaker._Underlying *_other);
                _UnderlyingPtr = __MR_Parallel_CallSimplyMaker_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Parallel::CallSimplyMaker::operator()`.
            public unsafe MR.Parallel.CallSimply Call()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_call", ExactSpelling = true)]
                extern static MR.Parallel.CallSimply._Underlying *__MR_Parallel_CallSimplyMaker_call(_Underlying *_this);
                return new(__MR_Parallel_CallSimplyMaker_call(_UnderlyingPtr), is_owning: true);
            }
        }

        /// Generated from class `MR::Parallel::CallSimplyMaker`.
        /// This is the non-const half of the class.
        public class CallSimplyMaker : Const_CallSimplyMaker
        {
            internal unsafe CallSimplyMaker(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe CallSimplyMaker() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Parallel.CallSimplyMaker._Underlying *__MR_Parallel_CallSimplyMaker_DefaultConstruct();
                _UnderlyingPtr = __MR_Parallel_CallSimplyMaker_DefaultConstruct();
            }

            /// Generated from constructor `MR::Parallel::CallSimplyMaker::CallSimplyMaker`.
            public unsafe CallSimplyMaker(MR.Parallel.Const_CallSimplyMaker _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimplyMaker._Underlying *__MR_Parallel_CallSimplyMaker_ConstructFromAnother(MR.Parallel.CallSimplyMaker._Underlying *_other);
                _UnderlyingPtr = __MR_Parallel_CallSimplyMaker_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Parallel::CallSimplyMaker::operator=`.
            public unsafe MR.Parallel.CallSimplyMaker Assign(MR.Parallel.Const_CallSimplyMaker _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parallel_CallSimplyMaker_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Parallel.CallSimplyMaker._Underlying *__MR_Parallel_CallSimplyMaker_AssignFromAnother(_Underlying *_this, MR.Parallel.CallSimplyMaker._Underlying *_other);
                return new(__MR_Parallel_CallSimplyMaker_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `CallSimplyMaker` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_CallSimplyMaker`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CallSimplyMaker`/`Const_CallSimplyMaker` directly.
        public class _InOptMut_CallSimplyMaker
        {
            public CallSimplyMaker? Opt;

            public _InOptMut_CallSimplyMaker() {}
            public _InOptMut_CallSimplyMaker(CallSimplyMaker value) {Opt = value;}
            public static implicit operator _InOptMut_CallSimplyMaker(CallSimplyMaker value) {return new(value);}
        }

        /// This is used for optional parameters of class `CallSimplyMaker` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_CallSimplyMaker`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CallSimplyMaker`/`Const_CallSimplyMaker` to pass it to the function.
        public class _InOptConst_CallSimplyMaker
        {
            public Const_CallSimplyMaker? Opt;

            public _InOptConst_CallSimplyMaker() {}
            public _InOptConst_CallSimplyMaker(Const_CallSimplyMaker value) {Opt = value;}
            public static implicit operator _InOptConst_CallSimplyMaker(Const_CallSimplyMaker value) {return new(value);}
        }
    }
}

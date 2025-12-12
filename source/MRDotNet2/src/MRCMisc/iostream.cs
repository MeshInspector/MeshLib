public static partial class MR
{
    public static partial class Std
    {
        /// This is the const half of the class.
        public class Const_Ostream : MR.Misc.Object
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Ostream(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}
        }

        /// This is the non-const half of the class.
        public class Ostream : Const_Ostream
        {
            internal unsafe Ostream(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Generated from function `MR::operator<<<float>`.
            public unsafe MR.Std.Ostream Lshift(MR.Const_Vector3f vec)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_print_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Std.Ostream._Underlying *__MR_print_MR_Vector3f(_Underlying *s, MR.Const_Vector3f._Underlying *vec);
                return new(__MR_print_MR_Vector3f(_UnderlyingPtr, vec._UnderlyingPtr), is_owning: false);
            }

            // =====================================================================
            // PointOnFace
            /// Generated from function `MR::operator<<`.
            public unsafe MR.Std.Ostream Lshift(MR.Const_PointOnFace pof)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_print_MR_PointOnFace", ExactSpelling = true)]
                extern static MR.Std.Ostream._Underlying *__MR_print_MR_PointOnFace(_Underlying *s, MR.Const_PointOnFace._Underlying *pof);
                return new(__MR_print_MR_PointOnFace(_UnderlyingPtr, pof._UnderlyingPtr), is_owning: false);
            }

            // =====================================================================
            // BitSet, format compatible with boost::dynamic_bitset
            /// Generated from function `MR::operator<<`.
            public unsafe MR.Std.Ostream Lshift(MR.Const_BitSet bs)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_print_MR_BitSet", ExactSpelling = true)]
                extern static MR.Std.Ostream._Underlying *__MR_print_MR_BitSet(_Underlying *s, MR.Const_BitSet._Underlying *bs);
                return new(__MR_print_MR_BitSet(_UnderlyingPtr, bs._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Ostream` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Ostream`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Ostream`/`Const_Ostream` directly.
        public class _InOptMut_Ostream
        {
            public Ostream? Opt;

            public _InOptMut_Ostream() {}
            public _InOptMut_Ostream(Ostream value) {Opt = value;}
            public static implicit operator _InOptMut_Ostream(Ostream value) {return new(value);}
        }

        /// This is used for optional parameters of class `Ostream` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Ostream`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Ostream`/`Const_Ostream` to pass it to the function.
        public class _InOptConst_Ostream
        {
            public Const_Ostream? Opt;

            public _InOptConst_Ostream() {}
            public _InOptConst_Ostream(Const_Ostream value) {Opt = value;}
            public static implicit operator _InOptConst_Ostream(Const_Ostream value) {return new(value);}
        }

        /// This is the const half of the class.
        public class Const_Istream : MR.Misc.Object
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Istream(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}
        }

        /// This is the non-const half of the class.
        public class Istream : Const_Istream
        {
            internal unsafe Istream(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Generated from function `MR::operator>><float>`.
            public unsafe MR.Std.Istream Rshift(MR.Mut_Vector3f vec)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_input_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Std.Istream._Underlying *__MR_input_MR_Vector3f(_Underlying *s, MR.Mut_Vector3f._Underlying *vec);
                return new(__MR_input_MR_Vector3f(_UnderlyingPtr, vec._UnderlyingPtr), is_owning: false);
            }

            /// Generated from function `MR::operator>>`.
            public unsafe MR.Std.Istream Rshift(MR.PointOnFace pof)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_input_MR_PointOnFace", ExactSpelling = true)]
                extern static MR.Std.Istream._Underlying *__MR_input_MR_PointOnFace(_Underlying *s, MR.PointOnFace._Underlying *pof);
                return new(__MR_input_MR_PointOnFace(_UnderlyingPtr, pof._UnderlyingPtr), is_owning: false);
            }

            /// Generated from function `MR::operator>>`.
            public unsafe MR.Std.Istream Rshift(MR.BitSet bs)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_input_MR_BitSet", ExactSpelling = true)]
                extern static MR.Std.Istream._Underlying *__MR_input_MR_BitSet(_Underlying *s, MR.BitSet._Underlying *bs);
                return new(__MR_input_MR_BitSet(_UnderlyingPtr, bs._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Istream` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Istream`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Istream`/`Const_Istream` directly.
        public class _InOptMut_Istream
        {
            public Istream? Opt;

            public _InOptMut_Istream() {}
            public _InOptMut_Istream(Istream value) {Opt = value;}
            public static implicit operator _InOptMut_Istream(Istream value) {return new(value);}
        }

        /// This is used for optional parameters of class `Istream` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Istream`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Istream`/`Const_Istream` to pass it to the function.
        public class _InOptConst_Istream
        {
            public Const_Istream? Opt;

            public _InOptConst_Istream() {}
            public _InOptConst_Istream(Const_Istream value) {Opt = value;}
            public static implicit operator _InOptConst_Istream(Const_Istream value) {return new(value);}
        }
    }

    /// Returns the `stdout` stream.
    public static unsafe MR.Std.Ostream GetStdCout()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetStdCout", ExactSpelling = true)]
        extern static MR.Std.Ostream._Underlying *__MR_GetStdCout();
        return new(__MR_GetStdCout(), is_owning: false);
    }

    /// Returns the `stderr` stream, buffered.
    public static unsafe MR.Std.Ostream GetStdCerr()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetStdCerr", ExactSpelling = true)]
        extern static MR.Std.Ostream._Underlying *__MR_GetStdCerr();
        return new(__MR_GetStdCerr(), is_owning: false);
    }

    /// Returns the `stderr` stream, unbuffered.
    public static unsafe MR.Std.Ostream GetStdClog()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetStdClog", ExactSpelling = true)]
        extern static MR.Std.Ostream._Underlying *__MR_GetStdClog();
        return new(__MR_GetStdClog(), is_owning: false);
    }

    /// Returns the `stdin` stream.
    public static unsafe MR.Std.Istream GetStdCin()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GetStdCin", ExactSpelling = true)]
        extern static MR.Std.Istream._Underlying *__MR_GetStdCin();
        return new(__MR_GetStdCin(), is_owning: false);
    }
}

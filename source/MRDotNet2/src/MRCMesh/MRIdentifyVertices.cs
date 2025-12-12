public static partial class MR
{
    public static partial class MeshBuilder
    {
        /// this makes bit-wise comparison of two Vector3f's thus making two NaNs equal
        /// Generated from class `MR::MeshBuilder::equalVector3f`.
        /// This is the const half of the class.
        public class Const_EqualVector3f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_EqualVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_equalVector3f_Destroy(_Underlying *_this);
                __MR_MeshBuilder_equalVector3f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_EqualVector3f() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_EqualVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.EqualVector3f._Underlying *__MR_MeshBuilder_equalVector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_equalVector3f_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::equalVector3f::equalVector3f`.
            public unsafe Const_EqualVector3f(MR.MeshBuilder.Const_EqualVector3f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.EqualVector3f._Underlying *__MR_MeshBuilder_equalVector3f_ConstructFromAnother(MR.MeshBuilder.EqualVector3f._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_equalVector3f_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::equalVector3f::operator()`.
            public unsafe bool Call(MR.Const_Vector3f a, MR.Const_Vector3f b)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_call", ExactSpelling = true)]
                extern static byte __MR_MeshBuilder_equalVector3f_call(_Underlying *_this, MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
                return __MR_MeshBuilder_equalVector3f_call(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr) != 0;
            }
        }

        /// this makes bit-wise comparison of two Vector3f's thus making two NaNs equal
        /// Generated from class `MR::MeshBuilder::equalVector3f`.
        /// This is the non-const half of the class.
        public class EqualVector3f : Const_EqualVector3f
        {
            internal unsafe EqualVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe EqualVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.EqualVector3f._Underlying *__MR_MeshBuilder_equalVector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_equalVector3f_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::equalVector3f::equalVector3f`.
            public unsafe EqualVector3f(MR.MeshBuilder.Const_EqualVector3f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.EqualVector3f._Underlying *__MR_MeshBuilder_equalVector3f_ConstructFromAnother(MR.MeshBuilder.EqualVector3f._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_equalVector3f_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::equalVector3f::operator=`.
            public unsafe MR.MeshBuilder.EqualVector3f Assign(MR.MeshBuilder.Const_EqualVector3f _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_equalVector3f_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.EqualVector3f._Underlying *__MR_MeshBuilder_equalVector3f_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.EqualVector3f._Underlying *_other);
                return new(__MR_MeshBuilder_equalVector3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `EqualVector3f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_EqualVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `EqualVector3f`/`Const_EqualVector3f` directly.
        public class _InOptMut_EqualVector3f
        {
            public EqualVector3f? Opt;

            public _InOptMut_EqualVector3f() {}
            public _InOptMut_EqualVector3f(EqualVector3f value) {Opt = value;}
            public static implicit operator _InOptMut_EqualVector3f(EqualVector3f value) {return new(value);}
        }

        /// This is used for optional parameters of class `EqualVector3f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_EqualVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `EqualVector3f`/`Const_EqualVector3f` to pass it to the function.
        public class _InOptConst_EqualVector3f
        {
            public Const_EqualVector3f? Opt;

            public _InOptConst_EqualVector3f() {}
            public _InOptConst_EqualVector3f(Const_EqualVector3f value) {Opt = value;}
            public static implicit operator _InOptConst_EqualVector3f(Const_EqualVector3f value) {return new(value);}
        }

        /// this class is responsible for giving a unique id to each vertex with distinct coordinates
        /// NOTE: the points are considered non-identical if they have the same values but have zero values with different signs
        /// (e.g. (0; 0; 1) and (-0; 0; 1))
        /// use `Vector3::unsignZeroValues` method to get rid of signed zero values if you're unsure of their absence
        /// Generated from class `MR::MeshBuilder::VertexIdentifier`.
        /// This is the const half of the class.
        public class Const_VertexIdentifier : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_VertexIdentifier(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_VertexIdentifier_Destroy(_Underlying *_this);
                __MR_MeshBuilder_VertexIdentifier_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_VertexIdentifier() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_VertexIdentifier() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertexIdentifier._Underlying *__MR_MeshBuilder_VertexIdentifier_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertexIdentifier_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::VertexIdentifier::VertexIdentifier`.
            public unsafe Const_VertexIdentifier(MR.MeshBuilder._ByValue_VertexIdentifier _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertexIdentifier._Underlying *__MR_MeshBuilder_VertexIdentifier_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.VertexIdentifier._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertexIdentifier_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// returns the number of triangles added so far
            /// Generated from method `MR::MeshBuilder::VertexIdentifier::numTris`.
            public unsafe ulong NumTris()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_numTris", ExactSpelling = true)]
                extern static ulong __MR_MeshBuilder_VertexIdentifier_numTris(_Underlying *_this);
                return __MR_MeshBuilder_VertexIdentifier_numTris(_UnderlyingPtr);
            }
        }

        /// this class is responsible for giving a unique id to each vertex with distinct coordinates
        /// NOTE: the points are considered non-identical if they have the same values but have zero values with different signs
        /// (e.g. (0; 0; 1) and (-0; 0; 1))
        /// use `Vector3::unsignZeroValues` method to get rid of signed zero values if you're unsure of their absence
        /// Generated from class `MR::MeshBuilder::VertexIdentifier`.
        /// This is the non-const half of the class.
        public class VertexIdentifier : Const_VertexIdentifier
        {
            internal unsafe VertexIdentifier(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe VertexIdentifier() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertexIdentifier._Underlying *__MR_MeshBuilder_VertexIdentifier_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertexIdentifier_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::VertexIdentifier::VertexIdentifier`.
            public unsafe VertexIdentifier(MR.MeshBuilder._ByValue_VertexIdentifier _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertexIdentifier._Underlying *__MR_MeshBuilder_VertexIdentifier_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.VertexIdentifier._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertexIdentifier_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshBuilder::VertexIdentifier::operator=`.
            public unsafe MR.MeshBuilder.VertexIdentifier Assign(MR.MeshBuilder._ByValue_VertexIdentifier _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertexIdentifier._Underlying *__MR_MeshBuilder_VertexIdentifier_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.VertexIdentifier._Underlying *_other);
                return new(__MR_MeshBuilder_VertexIdentifier_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }

            /// prepare identification of vertices from given this number of triangles
            /// Generated from method `MR::MeshBuilder::VertexIdentifier::reserve`.
            public unsafe void Reserve(ulong numTris)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_reserve", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_VertexIdentifier_reserve(_Underlying *_this, ulong numTris);
                __MR_MeshBuilder_VertexIdentifier_reserve(_UnderlyingPtr, numTris);
            }

            /// identifies vertices from a chunk of triangles
            /// Generated from method `MR::MeshBuilder::VertexIdentifier::addTriangles`.
            public unsafe void AddTriangles(MR.Std.Const_Vector_StdArrayMRVector3f3 buffer)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_addTriangles", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_VertexIdentifier_addTriangles(_Underlying *_this, MR.Std.Const_Vector_StdArrayMRVector3f3._Underlying *buffer);
                __MR_MeshBuilder_VertexIdentifier_addTriangles(_UnderlyingPtr, buffer._UnderlyingPtr);
            }

            /// obtains triangulation with vertex ids
            /// Generated from method `MR::MeshBuilder::VertexIdentifier::takeTriangulation`.
            public unsafe MR.Misc._Moved<MR.Triangulation> TakeTriangulation()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_takeTriangulation", ExactSpelling = true)]
                extern static MR.Triangulation._Underlying *__MR_MeshBuilder_VertexIdentifier_takeTriangulation(_Underlying *_this);
                return MR.Misc.Move(new MR.Triangulation(__MR_MeshBuilder_VertexIdentifier_takeTriangulation(_UnderlyingPtr), is_owning: true));
            }

            /// obtains coordinates of unique points in the order of vertex ids
            /// Generated from method `MR::MeshBuilder::VertexIdentifier::takePoints`.
            public unsafe MR.Misc._Moved<MR.VertCoords> TakePoints()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertexIdentifier_takePoints", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_MeshBuilder_VertexIdentifier_takePoints(_Underlying *_this);
                return MR.Misc.Move(new MR.VertCoords(__MR_MeshBuilder_VertexIdentifier_takePoints(_UnderlyingPtr), is_owning: true));
            }
        }

        /// This is used as a function parameter when the underlying function receives `VertexIdentifier` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `VertexIdentifier`/`Const_VertexIdentifier` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_VertexIdentifier
        {
            internal readonly Const_VertexIdentifier? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_VertexIdentifier() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_VertexIdentifier(Const_VertexIdentifier new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_VertexIdentifier(Const_VertexIdentifier arg) {return new(arg);}
            public _ByValue_VertexIdentifier(MR.Misc._Moved<VertexIdentifier> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_VertexIdentifier(MR.Misc._Moved<VertexIdentifier> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `VertexIdentifier` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_VertexIdentifier`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertexIdentifier`/`Const_VertexIdentifier` directly.
        public class _InOptMut_VertexIdentifier
        {
            public VertexIdentifier? Opt;

            public _InOptMut_VertexIdentifier() {}
            public _InOptMut_VertexIdentifier(VertexIdentifier value) {Opt = value;}
            public static implicit operator _InOptMut_VertexIdentifier(VertexIdentifier value) {return new(value);}
        }

        /// This is used for optional parameters of class `VertexIdentifier` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_VertexIdentifier`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertexIdentifier`/`Const_VertexIdentifier` to pass it to the function.
        public class _InOptConst_VertexIdentifier
        {
            public Const_VertexIdentifier? Opt;

            public _InOptConst_VertexIdentifier() {}
            public _InOptConst_VertexIdentifier(Const_VertexIdentifier value) {Opt = value;}
            public static implicit operator _InOptConst_VertexIdentifier(Const_VertexIdentifier value) {return new(value);}
        }
    }
}

public static partial class MR
{
    public static partial class MeshBuilder
    {
        /// Generated from class `MR::MeshBuilder::VertDuplication`.
        /// This is the const half of the class.
        public class Const_VertDuplication : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_VertDuplication(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_VertDuplication_Destroy(_Underlying *_this);
                __MR_MeshBuilder_VertDuplication_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_VertDuplication() {Dispose(false);}

            // original vertex before duplication
            public unsafe MR.Const_VertId SrcVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_Get_srcVert", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_MeshBuilder_VertDuplication_Get_srcVert(_Underlying *_this);
                    return new(__MR_MeshBuilder_VertDuplication_Get_srcVert(_UnderlyingPtr), is_owning: false);
                }
            }

            // new vertex after duplication
            public unsafe MR.Const_VertId DupVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_Get_dupVert", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_MeshBuilder_VertDuplication_Get_dupVert(_Underlying *_this);
                    return new(__MR_MeshBuilder_VertDuplication_Get_dupVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_VertDuplication() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::VertDuplication` elementwise.
            public unsafe Const_VertDuplication(MR.VertId srcVert, MR.VertId dupVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_ConstructFrom(MR.VertId srcVert, MR.VertId dupVert);
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_ConstructFrom(srcVert, dupVert);
            }

            /// Generated from constructor `MR::MeshBuilder::VertDuplication::VertDuplication`.
            public unsafe Const_VertDuplication(MR.MeshBuilder.Const_VertDuplication _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_ConstructFromAnother(MR.MeshBuilder.VertDuplication._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::MeshBuilder::VertDuplication`.
        /// This is the non-const half of the class.
        public class VertDuplication : Const_VertDuplication
        {
            internal unsafe VertDuplication(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // original vertex before duplication
            public new unsafe MR.Mut_VertId SrcVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_GetMutable_srcVert", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_MeshBuilder_VertDuplication_GetMutable_srcVert(_Underlying *_this);
                    return new(__MR_MeshBuilder_VertDuplication_GetMutable_srcVert(_UnderlyingPtr), is_owning: false);
                }
            }

            // new vertex after duplication
            public new unsafe MR.Mut_VertId DupVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_GetMutable_dupVert", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_MeshBuilder_VertDuplication_GetMutable_dupVert(_Underlying *_this);
                    return new(__MR_MeshBuilder_VertDuplication_GetMutable_dupVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe VertDuplication() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::VertDuplication` elementwise.
            public unsafe VertDuplication(MR.VertId srcVert, MR.VertId dupVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_ConstructFrom(MR.VertId srcVert, MR.VertId dupVert);
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_ConstructFrom(srcVert, dupVert);
            }

            /// Generated from constructor `MR::MeshBuilder::VertDuplication::VertDuplication`.
            public unsafe VertDuplication(MR.MeshBuilder.Const_VertDuplication _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_ConstructFromAnother(MR.MeshBuilder.VertDuplication._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertDuplication_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::VertDuplication::operator=`.
            public unsafe MR.MeshBuilder.VertDuplication Assign(MR.MeshBuilder.Const_VertDuplication _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertDuplication_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertDuplication._Underlying *__MR_MeshBuilder_VertDuplication_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.VertDuplication._Underlying *_other);
                return new(__MR_MeshBuilder_VertDuplication_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `VertDuplication` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_VertDuplication`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertDuplication`/`Const_VertDuplication` directly.
        public class _InOptMut_VertDuplication
        {
            public VertDuplication? Opt;

            public _InOptMut_VertDuplication() {}
            public _InOptMut_VertDuplication(VertDuplication value) {Opt = value;}
            public static implicit operator _InOptMut_VertDuplication(VertDuplication value) {return new(value);}
        }

        /// This is used for optional parameters of class `VertDuplication` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_VertDuplication`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertDuplication`/`Const_VertDuplication` to pass it to the function.
        public class _InOptConst_VertDuplication
        {
            public Const_VertDuplication? Opt;

            public _InOptConst_VertDuplication() {}
            public _InOptConst_VertDuplication(Const_VertDuplication value) {Opt = value;}
            public static implicit operator _InOptConst_VertDuplication(Const_VertDuplication value) {return new(value);}
        }

        // a part of a whole mesh to be constructed
        /// Generated from class `MR::MeshBuilder::MeshPiece`.
        /// This is the const half of the class.
        public class Const_MeshPiece : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_MeshPiece(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_MeshPiece_Destroy(_Underlying *_this);
                __MR_MeshBuilder_MeshPiece_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_MeshPiece() {Dispose(false);}

            // face of part -> face of whole mesh
            public unsafe MR.Const_FaceMap Fmap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_Get_fmap", ExactSpelling = true)]
                    extern static MR.Const_FaceMap._Underlying *__MR_MeshBuilder_MeshPiece_Get_fmap(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_Get_fmap(_UnderlyingPtr), is_owning: false);
                }
            }

            // vert of part -> vert of whole mesh
            public unsafe MR.Const_VertMap Vmap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_Get_vmap", ExactSpelling = true)]
                    extern static MR.Const_VertMap._Underlying *__MR_MeshBuilder_MeshPiece_Get_vmap(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_Get_vmap(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_MeshTopology Topology
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_Get_topology", ExactSpelling = true)]
                    extern static MR.Const_MeshTopology._Underlying *__MR_MeshBuilder_MeshPiece_Get_topology(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_Get_topology(_UnderlyingPtr), is_owning: false);
                }
            }

            // remaining triangles of part, not in topology
            public unsafe MR.Const_FaceBitSet Rem
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_Get_rem", ExactSpelling = true)]
                    extern static MR.Const_FaceBitSet._Underlying *__MR_MeshBuilder_MeshPiece_Get_rem(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_Get_rem(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_MeshPiece() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::MeshPiece` elementwise.
            public unsafe Const_MeshPiece(MR._ByValue_FaceMap fmap, MR._ByValue_VertMap vmap, MR._ByValue_MeshTopology topology, MR._ByValue_FaceBitSet rem) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_ConstructFrom(MR.Misc._PassBy fmap_pass_by, MR.FaceMap._Underlying *fmap, MR.Misc._PassBy vmap_pass_by, MR.VertMap._Underlying *vmap, MR.Misc._PassBy topology_pass_by, MR.MeshTopology._Underlying *topology, MR.Misc._PassBy rem_pass_by, MR.FaceBitSet._Underlying *rem);
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_ConstructFrom(fmap.PassByMode, fmap.Value is not null ? fmap.Value._UnderlyingPtr : null, vmap.PassByMode, vmap.Value is not null ? vmap.Value._UnderlyingPtr : null, topology.PassByMode, topology.Value is not null ? topology.Value._UnderlyingPtr : null, rem.PassByMode, rem.Value is not null ? rem.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshBuilder::MeshPiece::MeshPiece`.
            public unsafe Const_MeshPiece(MR.MeshBuilder._ByValue_MeshPiece _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.MeshPiece._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        // a part of a whole mesh to be constructed
        /// Generated from class `MR::MeshBuilder::MeshPiece`.
        /// This is the non-const half of the class.
        public class MeshPiece : Const_MeshPiece
        {
            internal unsafe MeshPiece(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // face of part -> face of whole mesh
            public new unsafe MR.FaceMap Fmap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_GetMutable_fmap", ExactSpelling = true)]
                    extern static MR.FaceMap._Underlying *__MR_MeshBuilder_MeshPiece_GetMutable_fmap(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_GetMutable_fmap(_UnderlyingPtr), is_owning: false);
                }
            }

            // vert of part -> vert of whole mesh
            public new unsafe MR.VertMap Vmap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_GetMutable_vmap", ExactSpelling = true)]
                    extern static MR.VertMap._Underlying *__MR_MeshBuilder_MeshPiece_GetMutable_vmap(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_GetMutable_vmap(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.MeshTopology Topology
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_GetMutable_topology", ExactSpelling = true)]
                    extern static MR.MeshTopology._Underlying *__MR_MeshBuilder_MeshPiece_GetMutable_topology(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_GetMutable_topology(_UnderlyingPtr), is_owning: false);
                }
            }

            // remaining triangles of part, not in topology
            public new unsafe MR.FaceBitSet Rem
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_GetMutable_rem", ExactSpelling = true)]
                    extern static MR.FaceBitSet._Underlying *__MR_MeshBuilder_MeshPiece_GetMutable_rem(_Underlying *_this);
                    return new(__MR_MeshBuilder_MeshPiece_GetMutable_rem(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe MeshPiece() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::MeshPiece` elementwise.
            public unsafe MeshPiece(MR._ByValue_FaceMap fmap, MR._ByValue_VertMap vmap, MR._ByValue_MeshTopology topology, MR._ByValue_FaceBitSet rem) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_ConstructFrom(MR.Misc._PassBy fmap_pass_by, MR.FaceMap._Underlying *fmap, MR.Misc._PassBy vmap_pass_by, MR.VertMap._Underlying *vmap, MR.Misc._PassBy topology_pass_by, MR.MeshTopology._Underlying *topology, MR.Misc._PassBy rem_pass_by, MR.FaceBitSet._Underlying *rem);
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_ConstructFrom(fmap.PassByMode, fmap.Value is not null ? fmap.Value._UnderlyingPtr : null, vmap.PassByMode, vmap.Value is not null ? vmap.Value._UnderlyingPtr : null, topology.PassByMode, topology.Value is not null ? topology.Value._UnderlyingPtr : null, rem.PassByMode, rem.Value is not null ? rem.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshBuilder::MeshPiece::MeshPiece`.
            public unsafe MeshPiece(MR.MeshBuilder._ByValue_MeshPiece _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.MeshPiece._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_MeshPiece_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshBuilder::MeshPiece::operator=`.
            public unsafe MR.MeshBuilder.MeshPiece Assign(MR.MeshBuilder._ByValue_MeshPiece _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_MeshPiece_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.MeshPiece._Underlying *__MR_MeshBuilder_MeshPiece_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshBuilder.MeshPiece._Underlying *_other);
                return new(__MR_MeshBuilder_MeshPiece_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `MeshPiece` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `MeshPiece`/`Const_MeshPiece` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_MeshPiece
        {
            internal readonly Const_MeshPiece? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_MeshPiece() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_MeshPiece(Const_MeshPiece new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_MeshPiece(Const_MeshPiece arg) {return new(arg);}
            public _ByValue_MeshPiece(MR.Misc._Moved<MeshPiece> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_MeshPiece(MR.Misc._Moved<MeshPiece> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `MeshPiece` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshPiece`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MeshPiece`/`Const_MeshPiece` directly.
        public class _InOptMut_MeshPiece
        {
            public MeshPiece? Opt;

            public _InOptMut_MeshPiece() {}
            public _InOptMut_MeshPiece(MeshPiece value) {Opt = value;}
            public static implicit operator _InOptMut_MeshPiece(MeshPiece value) {return new(value);}
        }

        /// This is used for optional parameters of class `MeshPiece` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshPiece`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MeshPiece`/`Const_MeshPiece` to pass it to the function.
        public class _InOptConst_MeshPiece
        {
            public Const_MeshPiece? Opt;

            public _InOptConst_MeshPiece() {}
            public _InOptConst_MeshPiece(Const_MeshPiece value) {Opt = value;}
            public static implicit operator _InOptConst_MeshPiece(Const_MeshPiece value) {return new(value);}
        }

        /// Generated from class `MR::MeshBuilder::UniteCloseParams`.
        /// This is the const half of the class.
        public class Const_UniteCloseParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_UniteCloseParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_UniteCloseParams_Destroy(_Underlying *_this);
                __MR_MeshBuilder_UniteCloseParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_UniteCloseParams() {Dispose(false);}

            public unsafe float CloseDist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_closeDist", ExactSpelling = true)]
                    extern static float *__MR_MeshBuilder_UniteCloseParams_Get_closeDist(_Underlying *_this);
                    return *__MR_MeshBuilder_UniteCloseParams_Get_closeDist(_UnderlyingPtr);
                }
            }

            public unsafe bool UniteOnlyBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_uniteOnlyBd", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_UniteCloseParams_Get_uniteOnlyBd(_Underlying *_this);
                    return *__MR_MeshBuilder_UniteCloseParams_Get_uniteOnlyBd(_UnderlyingPtr);
                }
            }

            public unsafe ref void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_region", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_Get_region(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_Get_region(_UnderlyingPtr);
                }
            }

            public unsafe bool DuplicateNonManifold
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_duplicateNonManifold", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_UniteCloseParams_Get_duplicateNonManifold(_Underlying *_this);
                    return *__MR_MeshBuilder_UniteCloseParams_Get_duplicateNonManifold(_UnderlyingPtr);
                }
            }

            public unsafe ref void * OptionalVertOldToNew
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_optionalVertOldToNew", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_Get_optionalVertOldToNew(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_Get_optionalVertOldToNew(_UnderlyingPtr);
                }
            }

            public unsafe ref void * OptionalDuplications
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_Get_optionalDuplications", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_Get_optionalDuplications(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_Get_optionalDuplications(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_UniteCloseParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::UniteCloseParams` elementwise.
            public unsafe Const_UniteCloseParams(float closeDist, bool uniteOnlyBd, MR.VertBitSet? region, bool duplicateNonManifold, MR.VertMap? optionalVertOldToNew, MR.Std.Vector_MRMeshBuilderVertDuplication? optionalDuplications) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_ConstructFrom(float closeDist, byte uniteOnlyBd, MR.VertBitSet._Underlying *region, byte duplicateNonManifold, MR.VertMap._Underlying *optionalVertOldToNew, MR.Std.Vector_MRMeshBuilderVertDuplication._Underlying *optionalDuplications);
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_ConstructFrom(closeDist, uniteOnlyBd ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, duplicateNonManifold ? (byte)1 : (byte)0, optionalVertOldToNew is not null ? optionalVertOldToNew._UnderlyingPtr : null, optionalDuplications is not null ? optionalDuplications._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshBuilder::UniteCloseParams::UniteCloseParams`.
            public unsafe Const_UniteCloseParams(MR.MeshBuilder.Const_UniteCloseParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_ConstructFromAnother(MR.MeshBuilder.UniteCloseParams._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::MeshBuilder::UniteCloseParams`.
        /// This is the non-const half of the class.
        public class UniteCloseParams : Const_UniteCloseParams
        {
            internal unsafe UniteCloseParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref float CloseDist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_closeDist", ExactSpelling = true)]
                    extern static float *__MR_MeshBuilder_UniteCloseParams_GetMutable_closeDist(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_closeDist(_UnderlyingPtr);
                }
            }

            public new unsafe ref bool UniteOnlyBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_uniteOnlyBd", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_UniteCloseParams_GetMutable_uniteOnlyBd(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_uniteOnlyBd(_UnderlyingPtr);
                }
            }

            public new unsafe ref void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_region", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_GetMutable_region(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_region(_UnderlyingPtr);
                }
            }

            public new unsafe ref bool DuplicateNonManifold
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_duplicateNonManifold", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_UniteCloseParams_GetMutable_duplicateNonManifold(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_duplicateNonManifold(_UnderlyingPtr);
                }
            }

            public new unsafe ref void * OptionalVertOldToNew
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_optionalVertOldToNew", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_GetMutable_optionalVertOldToNew(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_optionalVertOldToNew(_UnderlyingPtr);
                }
            }

            public new unsafe ref void * OptionalDuplications
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_GetMutable_optionalDuplications", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_UniteCloseParams_GetMutable_optionalDuplications(_Underlying *_this);
                    return ref *__MR_MeshBuilder_UniteCloseParams_GetMutable_optionalDuplications(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe UniteCloseParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::UniteCloseParams` elementwise.
            public unsafe UniteCloseParams(float closeDist, bool uniteOnlyBd, MR.VertBitSet? region, bool duplicateNonManifold, MR.VertMap? optionalVertOldToNew, MR.Std.Vector_MRMeshBuilderVertDuplication? optionalDuplications) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_ConstructFrom(float closeDist, byte uniteOnlyBd, MR.VertBitSet._Underlying *region, byte duplicateNonManifold, MR.VertMap._Underlying *optionalVertOldToNew, MR.Std.Vector_MRMeshBuilderVertDuplication._Underlying *optionalDuplications);
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_ConstructFrom(closeDist, uniteOnlyBd ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, duplicateNonManifold ? (byte)1 : (byte)0, optionalVertOldToNew is not null ? optionalVertOldToNew._UnderlyingPtr : null, optionalDuplications is not null ? optionalDuplications._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshBuilder::UniteCloseParams::UniteCloseParams`.
            public unsafe UniteCloseParams(MR.MeshBuilder.Const_UniteCloseParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_ConstructFromAnother(MR.MeshBuilder.UniteCloseParams._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_UniteCloseParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::UniteCloseParams::operator=`.
            public unsafe MR.MeshBuilder.UniteCloseParams Assign(MR.MeshBuilder.Const_UniteCloseParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_UniteCloseParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.UniteCloseParams._Underlying *__MR_MeshBuilder_UniteCloseParams_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.UniteCloseParams._Underlying *_other);
                return new(__MR_MeshBuilder_UniteCloseParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `UniteCloseParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_UniteCloseParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `UniteCloseParams`/`Const_UniteCloseParams` directly.
        public class _InOptMut_UniteCloseParams
        {
            public UniteCloseParams? Opt;

            public _InOptMut_UniteCloseParams() {}
            public _InOptMut_UniteCloseParams(UniteCloseParams value) {Opt = value;}
            public static implicit operator _InOptMut_UniteCloseParams(UniteCloseParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `UniteCloseParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_UniteCloseParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `UniteCloseParams`/`Const_UniteCloseParams` to pass it to the function.
        public class _InOptConst_UniteCloseParams
        {
            public Const_UniteCloseParams? Opt;

            public _InOptConst_UniteCloseParams() {}
            public _InOptConst_UniteCloseParams(Const_UniteCloseParams value) {Opt = value;}
            public static implicit operator _InOptConst_UniteCloseParams(Const_UniteCloseParams value) {return new(value);}
        }

        /// construct mesh topology from a set of triangles with given ids;
        /// if skippedTris is given then it receives all input triangles not added in the resulting topology
        /// Generated from function `MR::MeshBuilder::fromTriangles`.
        /// Parameter `settings` defaults to `{}`.
        /// Parameter `progressCb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.MeshTopology> FromTriangles(MR.Const_Triangulation t, MR.MeshBuilder.Const_BuildSettings? settings = null, MR.Std._ByValue_Function_BoolFuncFromFloat? progressCb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_fromTriangles", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshBuilder_fromTriangles(MR.Const_Triangulation._Underlying *t, MR.MeshBuilder.Const_BuildSettings._Underlying *settings, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            return MR.Misc.Move(new MR.MeshTopology(__MR_MeshBuilder_fromTriangles(t._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, progressCb is not null ? progressCb.PassByMode : MR.Misc._PassBy.default_arg, progressCb is not null && progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null), is_owning: true));
        }

        // resolve non-manifold vertices by creating duplicate vertices in the triangulation (which is modified)
        // `lastValidVert` is needed if `region` or `t` does not contain full mesh, then first duplicated vertex will have `lastValidVert+1` index
        // return number of duplicated vertices
        /// Generated from function `MR::MeshBuilder::duplicateNonManifoldVertices`.
        /// Parameter `lastValidVert` defaults to `{}`.
        public static unsafe ulong DuplicateNonManifoldVertices(MR.Triangulation t, MR.FaceBitSet? region = null, MR.Std.Vector_MRMeshBuilderVertDuplication? dups = null, MR._InOpt_VertId lastValidVert = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_duplicateNonManifoldVertices", ExactSpelling = true)]
            extern static ulong __MR_MeshBuilder_duplicateNonManifoldVertices(MR.Triangulation._Underlying *t, MR.FaceBitSet._Underlying *region, MR.Std.Vector_MRMeshBuilderVertDuplication._Underlying *dups, MR.VertId *lastValidVert);
            return __MR_MeshBuilder_duplicateNonManifoldVertices(t._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, dups is not null ? dups._UnderlyingPtr : null, lastValidVert.HasValue ? &lastValidVert.Object : null);
        }

        // construct mesh topology from a set of triangles with given ids;
        // unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices;
        // triangulation is modified to introduce duplicates
        /// Generated from function `MR::MeshBuilder::fromTrianglesDuplicatingNonManifoldVertices`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.MeshTopology> FromTrianglesDuplicatingNonManifoldVertices(MR.Triangulation t, MR.Std.Vector_MRMeshBuilderVertDuplication? dups = null, MR.MeshBuilder.Const_BuildSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_fromTrianglesDuplicatingNonManifoldVertices", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshBuilder_fromTrianglesDuplicatingNonManifoldVertices(MR.Triangulation._Underlying *t, MR.Std.Vector_MRMeshBuilderVertDuplication._Underlying *dups, MR.MeshBuilder.Const_BuildSettings._Underlying *settings);
            return MR.Misc.Move(new MR.MeshTopology(__MR_MeshBuilder_fromTrianglesDuplicatingNonManifoldVertices(t._UnderlyingPtr, dups is not null ? dups._UnderlyingPtr : null, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        // construct mesh from point triples;
        // all coinciding points are given the same VertId in the result
        /// Generated from function `MR::MeshBuilder::fromPointTriples`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromPointTriples(MR.Std.Const_Vector_StdArrayMRVector3f3 posTriples)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_fromPointTriples", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshBuilder_fromPointTriples(MR.Std.Const_Vector_StdArrayMRVector3f3._Underlying *posTriples);
            return MR.Misc.Move(new MR.Mesh(__MR_MeshBuilder_fromPointTriples(posTriples._UnderlyingPtr), is_owning: true));
        }

        // construct mesh topology in parallel from given disjoint mesh pieces (which do not have any shared vertex)
        // and some additional triangles (in settings) that join the pieces
        /// Generated from function `MR::MeshBuilder::fromDisjointMeshPieces`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.MeshTopology> FromDisjointMeshPieces(MR.Const_Triangulation t, MR.VertId maxVertId, MR.Std.Const_Vector_MRMeshBuilderMeshPiece pieces, MR.MeshBuilder.Const_BuildSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_fromDisjointMeshPieces", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshBuilder_fromDisjointMeshPieces(MR.Const_Triangulation._Underlying *t, MR.VertId maxVertId, MR.Std.Const_Vector_MRMeshBuilderMeshPiece._Underlying *pieces, MR.MeshBuilder.Const_BuildSettings._Underlying *settings);
            return MR.Misc.Move(new MR.MeshTopology(__MR_MeshBuilder_fromDisjointMeshPieces(t._UnderlyingPtr, maxVertId, pieces._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        // adds triangles in the existing topology, given face indecies must be free;
        // settings.region on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
        /// Generated from function `MR::MeshBuilder::addTriangles`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe void AddTriangles(MR.MeshTopology res, MR.Const_Triangulation t, MR.MeshBuilder.Const_BuildSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_addTriangles_MR_Triangulation", ExactSpelling = true)]
            extern static void __MR_MeshBuilder_addTriangles_MR_Triangulation(MR.MeshTopology._Underlying *res, MR.Const_Triangulation._Underlying *t, MR.MeshBuilder.Const_BuildSettings._Underlying *settings);
            __MR_MeshBuilder_addTriangles_MR_Triangulation(res._UnderlyingPtr, t._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null);
        }

        // adds triangles in the existing topology, auto selecting face ids for them;
        // vertTriples on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
        /// Generated from function `MR::MeshBuilder::addTriangles`.
        public static unsafe void AddTriangles(MR.MeshTopology res, MR.Std.Vector_MRVertId vertTriples, MR.FaceBitSet? createdFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_addTriangles_std_vector_MR_VertId", ExactSpelling = true)]
            extern static void __MR_MeshBuilder_addTriangles_std_vector_MR_VertId(MR.MeshTopology._Underlying *res, MR.Std.Vector_MRVertId._Underlying *vertTriples, MR.FaceBitSet._Underlying *createdFaces);
            __MR_MeshBuilder_addTriangles_std_vector_MR_VertId(res._UnderlyingPtr, vertTriples._UnderlyingPtr, createdFaces is not null ? createdFaces._UnderlyingPtr : null);
        }

        /// construct mesh topology from face soup, where each face can have arbitrary degree (not only triangles)
        /// Generated from function `MR::MeshBuilder::fromFaceSoup`.
        /// Parameter `settings` defaults to `{}`.
        /// Parameter `progressCb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.MeshTopology> FromFaceSoup(MR.Std.Const_Vector_MRVertId verts, MR.Const_Vector_MRMeshBuilderVertSpan_MRFaceId faces, MR.MeshBuilder.Const_BuildSettings? settings = null, MR.Std._ByValue_Function_BoolFuncFromFloat? progressCb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_fromFaceSoup", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshBuilder_fromFaceSoup(MR.Std.Const_Vector_MRVertId._Underlying *verts, MR.Const_Vector_MRMeshBuilderVertSpan_MRFaceId._Underlying *faces, MR.MeshBuilder.Const_BuildSettings._Underlying *settings, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            return MR.Misc.Move(new MR.MeshTopology(__MR_MeshBuilder_fromFaceSoup(verts._UnderlyingPtr, faces._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, progressCb is not null ? progressCb.PassByMode : MR.Misc._PassBy.default_arg, progressCb is not null && progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// the function finds groups of mesh vertices located closer to each other than \param closeDist, and unites such vertices in one;
        /// then the mesh is rebuilt from the remaining triangles
        /// \param optionalVertOldToNew is the mapping of vertices: before -> after
        /// \param uniteOnlyBd if true then only boundary vertices can be united, all internal vertices (even close ones) will remain
        /// \return the number of vertices united, 0 means no change in the mesh
        /// Generated from function `MR::MeshBuilder::uniteCloseVertices`.
        /// Parameter `uniteOnlyBd` defaults to `true`.
        public static unsafe int UniteCloseVertices(MR.Mesh mesh, float closeDist, bool? uniteOnlyBd = null, MR.VertMap? optionalVertOldToNew = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_uniteCloseVertices_4", ExactSpelling = true)]
            extern static int __MR_MeshBuilder_uniteCloseVertices_4(MR.Mesh._Underlying *mesh, float closeDist, byte *uniteOnlyBd, MR.VertMap._Underlying *optionalVertOldToNew);
            byte __deref_uniteOnlyBd = uniteOnlyBd.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_MeshBuilder_uniteCloseVertices_4(mesh._UnderlyingPtr, closeDist, uniteOnlyBd.HasValue ? &__deref_uniteOnlyBd : null, optionalVertOldToNew is not null ? optionalVertOldToNew._UnderlyingPtr : null);
        }

        /// the function finds groups of mesh vertices located closer to each other than \param params.closeDist, and unites such vertices in one;
        /// then the mesh is rebuilt from the remaining triangles
        /// \return the number of vertices united, 0 means no change in the mesh
        /// Generated from function `MR::MeshBuilder::uniteCloseVertices`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe int UniteCloseVertices(MR.Mesh mesh, MR.MeshBuilder.Const_UniteCloseParams? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_uniteCloseVertices_2", ExactSpelling = true)]
            extern static int __MR_MeshBuilder_uniteCloseVertices_2(MR.Mesh._Underlying *mesh, MR.MeshBuilder.Const_UniteCloseParams._Underlying *params_);
            return __MR_MeshBuilder_uniteCloseVertices_2(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null);
        }
    }
}

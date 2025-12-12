public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 4 objects: `std::monostate`, `MR::MeshTriPoint`, `MR::EdgePoint`, `MR::VertId`.
        /// This is the const half of the class.
        public class Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Destroy(_Underlying *_this);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother(MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Index(_Underlying *_this);
                return __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `std::monostate`.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Std.Monostate value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate();
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate();
            }

            /// Constructs the variant storing the element 1, of type `MR::MeshTriPoint`.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Const_MeshTriPoint value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint(MR.MeshTriPoint._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 2, of type `MR::EdgePoint`.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Const_EdgePoint value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint(MR.EdgePoint._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 3, of type `MR::VertId`.
            public unsafe Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.VertId value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId(MR.VertId value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId(value);
            }

            /// Returns the element 0, of type `std::monostate`, read-only. If it's not the active element, returns null.
            public unsafe MR.Std.Monostate? GetStdMonostate()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_std_monostate", ExactSpelling = true)]
                extern static bool __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_std_monostate(_Underlying *_this);
                return __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_std_monostate(_UnderlyingPtr) ? new MR.Std.Monostate() : null;
            }

            /// Returns the element 1, of type `MR::MeshTriPoint`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_MeshTriPoint? GetMRMeshTriPoint()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_MeshTriPoint", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_MeshTriPoint(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_MeshTriPoint(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_MeshTriPoint(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::EdgePoint`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_EdgePoint? GetMREdgePoint()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_EdgePoint", ExactSpelling = true)]
                extern static MR.Const_EdgePoint._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_EdgePoint(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_EdgePoint(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_EdgePoint(__ret, is_owning: false) : null;
            }

            /// Returns the element 3, of type `MR::VertId`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_VertId? GetMRVertId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_VertId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_VertId(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_Get_MR_VertId(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_VertId(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 4 objects: `std::monostate`, `MR::MeshTriPoint`, `MR::EdgePoint`, `MR::VertId`.
        /// This is the non-const half of the class.
        public class Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId : Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId
        {
            internal unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother(MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *other);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `std::monostate`.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Std.Monostate value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate();
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_std_monostate();
            }

            /// Constructs the variant storing the element 1, of type `MR::MeshTriPoint`.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Const_MeshTriPoint value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint(MR.MeshTriPoint._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_MeshTriPoint(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 2, of type `MR::EdgePoint`.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.Const_EdgePoint value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint(MR.EdgePoint._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_EdgePoint(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 3, of type `MR::VertId`.
            public unsafe Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(MR.VertId value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId", ExactSpelling = true)]
                extern static MR.Std.Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId(MR.VertId value);
                _UnderlyingPtr = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_ConstructAs_MR_VertId(value);
            }

            /// Assigns to the variant, making it store the element 0, of type `std::monostate`.
            public unsafe void AssignAsStdMonostate(MR.Std.Monostate value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_std_monostate", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_std_monostate(_Underlying *_this);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_std_monostate(_UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 1, of type `MR::MeshTriPoint`.
            public unsafe void AssignAsMRMeshTriPoint(MR.Const_MeshTriPoint value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_MeshTriPoint", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_MeshTriPoint(_Underlying *_this, MR.MeshTriPoint._Underlying *value);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_MeshTriPoint(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 2, of type `MR::EdgePoint`.
            public unsafe void AssignAsMREdgePoint(MR.Const_EdgePoint value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_EdgePoint", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_EdgePoint(_Underlying *_this, MR.EdgePoint._Underlying *value);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_EdgePoint(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 3, of type `MR::VertId`.
            public unsafe void AssignAsMRVertId(MR.VertId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_VertId", ExactSpelling = true)]
                extern static void __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_VertId(_Underlying *_this, MR.VertId value);
                __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_AssignAs_MR_VertId(_UnderlyingPtr, value);
            }

            /// Returns the element 1, of type `MR::MeshTriPoint`, mutable. If it's not the active element, returns null.
            public unsafe MR.MeshTriPoint? GetMutableMRMeshTriPoint()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_MeshTriPoint", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_MeshTriPoint(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_MeshTriPoint(_UnderlyingPtr);
                return __ret is not null ? new MR.MeshTriPoint(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::EdgePoint`, mutable. If it's not the active element, returns null.
            public unsafe MR.EdgePoint? GetMutableMREdgePoint()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_EdgePoint", ExactSpelling = true)]
                extern static MR.EdgePoint._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_EdgePoint(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_EdgePoint(_UnderlyingPtr);
                return __ret is not null ? new MR.EdgePoint(__ret, is_owning: false) : null;
            }

            /// Returns the element 3, of type `MR::VertId`, mutable. If it's not the active element, returns null.
            public unsafe MR.Mut_VertId? GetMutableMRVertId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_VertId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_VertId(_Underlying *_this);
                var __ret = __MR_std_variant_std_monostate_MR_MeshTriPoint_MR_EdgePoint_MR_VertId_GetMutable_MR_VertId(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_VertId(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId`/`Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId` directly.
        public class _InOptMut_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId
        {
            public Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId? Opt;

            public _InOptMut_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId() {}
            public _InOptMut_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId`/`Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId` to pass it to the function.
        public class _InOptConst_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId
        {
            public Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId? Opt;

            public _InOptConst_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId() {}
            public _InOptConst_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId(Const_Variant_StdMonostate_MRMeshTriPoint_MREdgePoint_MRVertId value) {return new(value);}
        }
    }
}

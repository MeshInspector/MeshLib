public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `MR::FaceMap`, `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`.
        /// This is the const half of the class.
        public class Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Destroy(_Underlying *_this);
                __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Std._ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Index(_Underlying *_this);
                return __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::FaceMap`.
            public unsafe Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR._ByValue_FaceMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap(MR.Misc._PassBy value_pass_by, MR.FaceMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`.
            public unsafe Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Phmap._ByValue_FlatHashMap_MRFaceId_MRFaceId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::FaceMap`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_FaceMap? GetMRFaceMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_MR_FaceMap", ExactSpelling = true)]
                extern static MR.Const_FaceMap._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_MR_FaceMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_MR_FaceMap(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_FaceMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId? GetPhmapFlatHashMapMRFaceIdMRFaceId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
                extern static MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_Get_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `MR::FaceMap`, `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`.
        /// This is the non-const half of the class.
        public class Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId : Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId
        {
            internal unsafe Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Std._ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *other);
                __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::FaceMap`.
            public unsafe Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR._ByValue_FaceMap value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap(MR.Misc._PassBy value_pass_by, MR.FaceMap._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_MR_FaceMap(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`.
            public unsafe Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Phmap._ByValue_FlatHashMap_MRFaceId_MRFaceId value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
                extern static MR.Std.Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_ConstructAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::FaceMap`.
            public unsafe void AssignAsMRFaceMap(MR._ByValue_FaceMap value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_MR_FaceMap", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_MR_FaceMap(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.FaceMap._Underlying *value);
                __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_MR_FaceMap(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 1, of type `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`.
            public unsafe void AssignAsPhmapFlatHashMapMRFaceIdMRFaceId(MR.Phmap._ByValue_FlatHashMap_MRFaceId_MRFaceId value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *value);
                __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_AssignAs_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::FaceMap`, mutable. If it's not the active element, returns null.
            public unsafe MR.FaceMap? GetMutableMRFaceMap()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_MR_FaceMap", ExactSpelling = true)]
                extern static MR.FaceMap._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_MR_FaceMap(_Underlying *_this);
                var __ret = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_MR_FaceMap(_UnderlyingPtr);
                return __ret is not null ? new MR.FaceMap(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `phmap::flat_hash_map<MR::FaceId, MR::FaceId>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? GetMutablePhmapFlatHashMapMRFaceIdMRFaceId()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *__MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_Underlying *_this);
                var __ret = __MR_std_variant_MR_FaceMap_phmap_flat_hash_map_MR_FaceId_MR_FaceId_GetMutable_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_UnderlyingPtr);
                return __ret is not null ? new MR.Phmap.FlatHashMap_MRFaceId_MRFaceId(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId`/`Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId
        {
            internal readonly Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId arg) {return new(arg);}
            public _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Misc._Moved<Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(MR.Misc._Moved<Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId`/`Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` directly.
        public class _InOptMut_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId
        {
            public Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId? Opt;

            public _InOptMut_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() {}
            public _InOptMut_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId`/`Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId` to pass it to the function.
        public class _InOptConst_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId
        {
            public Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId? Opt;

            public _InOptConst_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId() {}
            public _InOptConst_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId(Const_Variant_MRFaceMap_PhmapFlatHashMapMRFaceIdMRFaceId value) {return new(value);}
        }
    }
}

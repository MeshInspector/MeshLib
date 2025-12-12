public static partial class MR
{
    public static partial class MeshSave
    {
        /// saves a number of named meshes in .obj file
        /// Generated from class `MR::MeshSave::NamedXfMesh`.
        /// This is the const half of the class.
        public class Const_NamedXfMesh : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_NamedXfMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshSave_NamedXfMesh_Destroy(_Underlying *_this);
                __MR_MeshSave_NamedXfMesh_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_NamedXfMesh() {Dispose(false);}

            public unsafe MR.Std.Const_String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_MeshSave_NamedXfMesh_Get_name(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_AffineXf3f ToWorld
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_Get_toWorld", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_MeshSave_NamedXfMesh_Get_toWorld(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_Get_toWorld(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Mesh Mesh
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_Get_mesh", ExactSpelling = true)]
                    extern static MR.Const_Mesh._UnderlyingShared *__MR_MeshSave_NamedXfMesh_Get_mesh(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_Get_mesh(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_NamedXfMesh() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_DefaultConstruct();
            }

            /// Constructs `MR::MeshSave::NamedXfMesh` elementwise.
            public unsafe Const_NamedXfMesh(ReadOnlySpan<char> name, MR.AffineXf3f toWorld, MR._ByValue_Mesh mesh) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_ConstructFrom(byte *name, byte *name_end, MR.AffineXf3f toWorld, MR.Misc._PassBy mesh_pass_by, MR.Mesh._UnderlyingShared *mesh);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_ConstructFrom(__ptr_name, __ptr_name + __len_name, toWorld, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingSharedPtr : null);
                }
            }

            /// Generated from constructor `MR::MeshSave::NamedXfMesh::NamedXfMesh`.
            public unsafe Const_NamedXfMesh(MR.MeshSave._ByValue_NamedXfMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshSave.NamedXfMesh._Underlying *_other);
                _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// saves a number of named meshes in .obj file
        /// Generated from class `MR::MeshSave::NamedXfMesh`.
        /// This is the non-const half of the class.
        public class NamedXfMesh : Const_NamedXfMesh
        {
            internal unsafe NamedXfMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_MeshSave_NamedXfMesh_GetMutable_name(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_AffineXf3f ToWorld
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_GetMutable_toWorld", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_MeshSave_NamedXfMesh_GetMutable_toWorld(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_GetMutable_toWorld(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mesh Mesh
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_GetMutable_mesh", ExactSpelling = true)]
                    extern static MR.Mesh._UnderlyingShared *__MR_MeshSave_NamedXfMesh_GetMutable_mesh(_Underlying *_this);
                    return new(__MR_MeshSave_NamedXfMesh_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe NamedXfMesh() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_DefaultConstruct();
            }

            /// Constructs `MR::MeshSave::NamedXfMesh` elementwise.
            public unsafe NamedXfMesh(ReadOnlySpan<char> name, MR.AffineXf3f toWorld, MR._ByValue_Mesh mesh) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_ConstructFrom(byte *name, byte *name_end, MR.AffineXf3f toWorld, MR.Misc._PassBy mesh_pass_by, MR.Mesh._UnderlyingShared *mesh);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_ConstructFrom(__ptr_name, __ptr_name + __len_name, toWorld, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingSharedPtr : null);
                }
            }

            /// Generated from constructor `MR::MeshSave::NamedXfMesh::NamedXfMesh`.
            public unsafe NamedXfMesh(MR.MeshSave._ByValue_NamedXfMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshSave.NamedXfMesh._Underlying *_other);
                _UnderlyingPtr = __MR_MeshSave_NamedXfMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshSave::NamedXfMesh::operator=`.
            public unsafe MR.MeshSave.NamedXfMesh Assign(MR.MeshSave._ByValue_NamedXfMesh _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_NamedXfMesh_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.NamedXfMesh._Underlying *__MR_MeshSave_NamedXfMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshSave.NamedXfMesh._Underlying *_other);
                return new(__MR_MeshSave_NamedXfMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `NamedXfMesh` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `NamedXfMesh`/`Const_NamedXfMesh` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_NamedXfMesh
        {
            internal readonly Const_NamedXfMesh? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_NamedXfMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_NamedXfMesh(Const_NamedXfMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_NamedXfMesh(Const_NamedXfMesh arg) {return new(arg);}
            public _ByValue_NamedXfMesh(MR.Misc._Moved<NamedXfMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_NamedXfMesh(MR.Misc._Moved<NamedXfMesh> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `NamedXfMesh` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_NamedXfMesh`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedXfMesh`/`Const_NamedXfMesh` directly.
        public class _InOptMut_NamedXfMesh
        {
            public NamedXfMesh? Opt;

            public _InOptMut_NamedXfMesh() {}
            public _InOptMut_NamedXfMesh(NamedXfMesh value) {Opt = value;}
            public static implicit operator _InOptMut_NamedXfMesh(NamedXfMesh value) {return new(value);}
        }

        /// This is used for optional parameters of class `NamedXfMesh` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_NamedXfMesh`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedXfMesh`/`Const_NamedXfMesh` to pass it to the function.
        public class _InOptConst_NamedXfMesh
        {
            public Const_NamedXfMesh? Opt;

            public _InOptConst_NamedXfMesh() {}
            public _InOptConst_NamedXfMesh(Const_NamedXfMesh value) {Opt = value;}
            public static implicit operator _InOptConst_NamedXfMesh(Const_NamedXfMesh value) {return new(value);}
        }

        /// Generated from function `MR::MeshSave::sceneToObj`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SceneToObj(MR.Std.Const_Vector_MRMeshSaveNamedXfMesh objects, ReadOnlySpan<char> file, MR.VertColors? colors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_sceneToObj_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_sceneToObj_std_filesystem_path(MR.Std.Const_Vector_MRMeshSaveNamedXfMesh._Underlying *objects, byte *file, byte *file_end, MR.VertColors._Underlying *colors);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_sceneToObj_std_filesystem_path(objects._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, colors is not null ? colors._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::sceneToObj`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SceneToObj(MR.Std.Const_Vector_MRMeshSaveNamedXfMesh objects, MR.Std.Ostream out_, MR.VertColors? colors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_sceneToObj_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_sceneToObj_std_ostream(MR.Std.Const_Vector_MRMeshSaveNamedXfMesh._Underlying *objects, MR.Std.Ostream._Underlying *out_, MR.VertColors._Underlying *colors);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_sceneToObj_std_ostream(objects._UnderlyingPtr, out_._UnderlyingPtr, colors is not null ? colors._UnderlyingPtr : null), is_owning: true));
        }
    }
}

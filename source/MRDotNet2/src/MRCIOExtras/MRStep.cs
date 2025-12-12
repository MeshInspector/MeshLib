public static partial class MR
{
    public static partial class MeshLoad
    {
        /// load mesh data from STEP file using OpenCASCADE
        /// Generated from function `MR::MeshLoad::fromStep`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromStep(ReadOnlySpan<char> path, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromStep_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromStep_std_filesystem_path(byte *path, byte *path_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromStep_std_filesystem_path(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshLoad::fromStep`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromStep(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromStep_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromStep_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromStep_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// load scene from STEP file using OpenCASCADE
        /// Generated from function `MR::MeshLoad::fromSceneStepFile`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRObject_StdString> FromSceneStepFile(ReadOnlySpan<char> path, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromSceneStepFile_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_StdSharedPtrMRObject_StdString._Underlying *__MR_MeshLoad_fromSceneStepFile_std_filesystem_path(byte *path, byte *path_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_StdSharedPtrMRObject_StdString(__MR_MeshLoad_fromSceneStepFile_std_filesystem_path(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshLoad::fromSceneStepFile`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRObject_StdString> FromSceneStepFile(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromSceneStepFile_std_istream", ExactSpelling = true)]
            extern static MR.Expected_StdSharedPtrMRObject_StdString._Underlying *__MR_MeshLoad_fromSceneStepFile_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_StdSharedPtrMRObject_StdString(__MR_MeshLoad_fromSceneStepFile_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }
}

public static partial class MR
{
    public static partial class ObjectSave
    {
        /// save an object tree to a given file
        /// file format must be scene-capable
        /// Generated from function `MR::ObjectSave::toAnySupportedSceneFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedSceneFormat(MR.Const_Object object_, ReadOnlySpan<char> file, MR.ObjectSave.Const_Settings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_toAnySupportedSceneFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ObjectSave_toAnySupportedSceneFormat(MR.Const_Object._Underlying *object_, byte *file, byte *file_end, MR.ObjectSave.Const_Settings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ObjectSave_toAnySupportedSceneFormat(object_._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// save a scene object to a given file
        /// if the file format is scene-capable, saves all the object's entities
        /// otherwise, saves only merged entities of the corresponding type (mesh, polyline, point cloud, etc.)
        /// Generated from function `MR::ObjectSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_Object object_, ReadOnlySpan<char> file, MR.ObjectSave.Const_Settings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_toAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_ObjectSave_toAnySupportedFormat(MR.Const_Object._Underlying *object_, byte *file, byte *file_end, MR.ObjectSave.Const_Settings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_ObjectSave_toAnySupportedFormat(object_._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }

    /**
    * \brief saves object subtree in given scene file (zip/mru)
    * \details format specification:
    *  children are saved under folder with name of their parent object
    *  all objects parameters are saved in one JSON file in the root folder
    *
    * if preCompress is set, it is called before compression
    * saving is controlled with Object::serializeModel_ and Object::serializeFields_
    */
    /// Generated from function `MR::serializeObjectTree`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SerializeObjectTree(MR.Const_Object object_, ReadOnlySpan<char> path, MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef preCompress, MR.ObjectSave.Const_Settings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_serializeObjectTree_4", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_serializeObjectTree_4(MR.Const_Object._Underlying *object_, byte *path, byte *path_end, MR.Misc._PassBy preCompress_pass_by, MR.Std.Function_VoidFuncFromConstStdFilesystemPathRef._Underlying *preCompress, MR.ObjectSave.Const_Settings._Underlying *settings);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_serializeObjectTree_4(object_._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, preCompress.PassByMode, preCompress.Value is not null ? preCompress.Value._UnderlyingPtr : null, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// Generated from function `MR::serializeObjectTree`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SerializeObjectTree(MR.Const_Object object_, ReadOnlySpan<char> path, MR.ObjectSave.Const_Settings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_serializeObjectTree_3", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_serializeObjectTree_3(MR.Const_Object._Underlying *object_, byte *path, byte *path_end, MR.ObjectSave.Const_Settings._Underlying *settings);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_serializeObjectTree_3(object_._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }
}

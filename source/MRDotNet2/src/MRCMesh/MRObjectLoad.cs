public static partial class MR
{
    /// loads mesh from given file in new object
    /// Generated from function `MR::makeObjectMeshFromFile`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectTMRObjectMesh_StdString> MakeObjectMeshFromFile(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectMeshFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectTMRObjectMesh_StdString._Underlying *__MR_makeObjectMeshFromFile(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectTMRObjectMesh_StdString(__MR_makeObjectMeshFromFile(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads data from given file and makes either ObjectMesh, ObjectLines or ObjectPoints (if the file has points or edges but not faces)
    /// Generated from function `MR::makeObjectFromMeshFile`.
    /// Parameter `cb` defaults to `{}`.
    /// Parameter `returnOnlyMesh` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> MakeObjectFromMeshFile(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null, bool? returnOnlyMesh = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectFromMeshFile", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_makeObjectFromMeshFile(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb, byte *returnOnlyMesh);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            byte __deref_returnOnlyMesh = returnOnlyMesh.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_makeObjectFromMeshFile(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null, returnOnlyMesh.HasValue ? &__deref_returnOnlyMesh : null), is_owning: true));
        }
    }

    /// loads lines from given file in new object
    /// Generated from function `MR::makeObjectLinesFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRObjectLines_StdString> MakeObjectLinesFromFile(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectLinesFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRObjectLines_StdString._Underlying *__MR_makeObjectLinesFromFile(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRObjectLines_StdString(__MR_makeObjectLinesFromFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads points from given file in new object
    /// Generated from function `MR::makeObjectPointsFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRObjectPoints_StdString> MakeObjectPointsFromFile(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectPointsFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRObjectPoints_StdString._Underlying *__MR_makeObjectPointsFromFile(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRObjectPoints_StdString(__MR_makeObjectPointsFromFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads distance map from given file in new object
    /// Generated from function `MR::makeObjectDistanceMapFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRObjectDistanceMap_StdString> MakeObjectDistanceMapFromFile(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectDistanceMapFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRObjectDistanceMap_StdString._Underlying *__MR_makeObjectDistanceMapFromFile(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRObjectDistanceMap_StdString(__MR_makeObjectDistanceMapFromFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// loads gcode from given file in new object
    /// Generated from function `MR::makeObjectGcodeFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRObjectGcode_StdString> MakeObjectGcodeFromFile(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectGcodeFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRObjectGcode_StdString._Underlying *__MR_makeObjectGcodeFromFile(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRObjectGcode_StdString(__MR_makeObjectGcodeFromFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /**
    * \brief load all objects (or any type: mesh, lines, points, voxels or scene) from file
    * \param callback - callback function to set progress (for progress bar)
    */
    /// Generated from function `MR::loadObjectFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjects_StdString> LoadObjectFromFile(ReadOnlySpan<char> filename, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadObjectFromFile", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjects_StdString._Underlying *__MR_loadObjectFromFile(byte *filename, byte *filename_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_filename = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(filename.Length)];
        int __len_filename = System.Text.Encoding.UTF8.GetBytes(filename, __bytes_filename);
        fixed (byte *__ptr_filename = __bytes_filename)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjects_StdString(__MR_loadObjectFromFile(__ptr_filename, __ptr_filename + __len_filename, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }

    // check if there are any supported files folder and subfolders
    /// Generated from function `MR::isSupportedFileInSubfolders`.
    public static unsafe bool IsSupportedFileInSubfolders(ReadOnlySpan<char> folder)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isSupportedFileInSubfolders", ExactSpelling = true)]
        extern static byte __MR_isSupportedFileInSubfolders(byte *folder, byte *folder_end);
        byte[] __bytes_folder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(folder.Length)];
        int __len_folder = System.Text.Encoding.UTF8.GetBytes(folder, __bytes_folder);
        fixed (byte *__ptr_folder = __bytes_folder)
        {
            return __MR_isSupportedFileInSubfolders(__ptr_folder, __ptr_folder + __len_folder) != 0;
        }
    }

    //tries to load scene from every format listed in SceneFormatFilters
    /// Generated from function `MR::loadSceneFromAnySupportedFormat`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> LoadSceneFromAnySupportedFormat(ReadOnlySpan<char> path, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadSceneFromAnySupportedFormat", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_loadSceneFromAnySupportedFormat(byte *path, byte *path_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_loadSceneFromAnySupportedFormat(__ptr_path, __ptr_path + __len_path, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }

    /**
    * \brief loads objects tree from given scene file (zip/mru)
    * \details format specification:
    *  children are saved under folder with name of their parent object
    *  all objects parameters are saved in one JSON file in the root folder
    *
    * if postDecompress is set, it is called after decompression
    * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
    */
    /// Generated from function `MR::deserializeObjectTree`.
    /// Parameter `postDecompress` defaults to `{}`.
    /// Parameter `progressCb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> DeserializeObjectTree(ReadOnlySpan<char> path, MR.Std.Const_Function_VoidFuncFromConstStdFilesystemPathRef? postDecompress = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deserializeObjectTree", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_deserializeObjectTree(byte *path, byte *path_end, MR.Std.Const_Function_VoidFuncFromConstStdFilesystemPathRef._Underlying *postDecompress, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCb);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_deserializeObjectTree(__ptr_path, __ptr_path + __len_path, postDecompress is not null ? postDecompress._UnderlyingPtr : null, progressCb is not null ? progressCb._UnderlyingPtr : null), is_owning: true));
        }
    }

    /**
    * \brief loads objects tree from given scene folder
    * \details format specification:
    *  children are saved under folder with name of their parent object
    *  all objects parameters are saved in one JSON file in the root folder
    *
    * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
    */
    /// Generated from function `MR::deserializeObjectTreeFromFolder`.
    /// Parameter `progressCb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> DeserializeObjectTreeFromFolder(ReadOnlySpan<char> folder, MR.Std.Const_Function_BoolFuncFromFloat? progressCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deserializeObjectTreeFromFolder", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_deserializeObjectTreeFromFolder(byte *folder, byte *folder_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCb);
        byte[] __bytes_folder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(folder.Length)];
        int __len_folder = System.Text.Encoding.UTF8.GetBytes(folder, __bytes_folder);
        fixed (byte *__ptr_folder = __bytes_folder)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_deserializeObjectTreeFromFolder(__ptr_folder, __ptr_folder + __len_folder, progressCb is not null ? progressCb._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// returns filters for all supported file formats for all types of objects
    /// Generated from function `MR::getAllFilters`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRIOFilter> GetAllFilters()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getAllFilters", ExactSpelling = true)]
        extern static MR.Std.Vector_MRIOFilter._Underlying *__MR_getAllFilters();
        return MR.Misc.Move(new MR.Std.Vector_MRIOFilter(__MR_getAllFilters(), is_owning: true));
    }
}

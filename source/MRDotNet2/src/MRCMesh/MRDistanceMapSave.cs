public static partial class MR
{
    public static partial class DistanceMapSave
    {
        /**
        * @brief Save DistanceMap to binary file
        * Format:
        * 2 integer - DistanceMap.resX & DistanceMap.resY
        * [resX * resY] float - matrix of values
        */
        /// Generated from function `MR::DistanceMapSave::toRAW`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToRAW(MR.Const_DistanceMap dmap, ReadOnlySpan<char> path, MR.Const_DistanceMapSaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSave_toRAW", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_DistanceMapSave_toRAW(MR.Const_DistanceMap._Underlying *dmap, byte *path, byte *path_end, MR.Const_DistanceMapSaveSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_DistanceMapSave_toRAW(dmap._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::DistanceMapSave::toMrDistanceMap`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToMrDistanceMap(MR.Const_DistanceMap dmap, ReadOnlySpan<char> path, MR.Const_DistanceMapSaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSave_toMrDistanceMap", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_DistanceMapSave_toMrDistanceMap(MR.Const_DistanceMap._Underlying *dmap, byte *path, byte *path_end, MR.Const_DistanceMapSaveSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_DistanceMapSave_toMrDistanceMap(dmap._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::DistanceMapSave::toAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_DistanceMap dmap, ReadOnlySpan<char> path, MR.Const_DistanceMapSaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSave_toAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_DistanceMapSave_toAnySupportedFormat(MR.Const_DistanceMap._Underlying *dmap, byte *path, byte *path_end, MR.Const_DistanceMapSaveSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_DistanceMapSave_toAnySupportedFormat(dmap._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

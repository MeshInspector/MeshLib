public static partial class MR
{
    public static partial class DistanceMapLoad
    {
        /**
        * @brief Load DistanceMap from binary file
        * Format:
        * 2 integer - DistanceMap.resX & DistanceMap.resY
        * [resX * resY] float - matrix of values
        */
        /// Generated from function `MR::DistanceMapLoad::fromRaw`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRDistanceMap_StdString> FromRaw(ReadOnlySpan<char> path, MR.Const_DistanceMapLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoad_fromRaw", ExactSpelling = true)]
            extern static MR.Expected_MRDistanceMap_StdString._Underlying *__MR_DistanceMapLoad_fromRaw(byte *path, byte *path_end, MR.Const_DistanceMapLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRDistanceMap_StdString(__MR_DistanceMapLoad_fromRaw(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::DistanceMapLoad::fromMrDistanceMap`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRDistanceMap_StdString> FromMrDistanceMap(ReadOnlySpan<char> path, MR.Const_DistanceMapLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoad_fromMrDistanceMap", ExactSpelling = true)]
            extern static MR.Expected_MRDistanceMap_StdString._Underlying *__MR_DistanceMapLoad_fromMrDistanceMap(byte *path, byte *path_end, MR.Const_DistanceMapLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRDistanceMap_StdString(__MR_DistanceMapLoad_fromMrDistanceMap(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::DistanceMapLoad::fromTiff`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRDistanceMap_StdString> FromTiff(ReadOnlySpan<char> path, MR.Const_DistanceMapLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoad_fromTiff", ExactSpelling = true)]
            extern static MR.Expected_MRDistanceMap_StdString._Underlying *__MR_DistanceMapLoad_fromTiff(byte *path, byte *path_end, MR.Const_DistanceMapLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRDistanceMap_StdString(__MR_DistanceMapLoad_fromTiff(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::DistanceMapLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRDistanceMap_StdString> FromAnySupportedFormat(ReadOnlySpan<char> path, MR.Const_DistanceMapLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoad_fromAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_MRDistanceMap_StdString._Underlying *__MR_DistanceMapLoad_fromAnySupportedFormat(byte *path, byte *path_end, MR.Const_DistanceMapLoadSettings._Underlying *settings);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRDistanceMap_StdString(__MR_DistanceMapLoad_fromAnySupportedFormat(__ptr_path, __ptr_path + __len_path, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

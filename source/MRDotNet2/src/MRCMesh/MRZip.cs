public static partial class MR
{
    /**
    * \brief decompresses given zip-file into given folder
    * \param password if password is given then it will be used to decipher encrypted archive
    */
    /// Generated from function `MR::decompressZip`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> DecompressZip(ReadOnlySpan<char> zipFile, ReadOnlySpan<char> targetFolder, byte? password = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decompressZip_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_decompressZip_std_filesystem_path(byte *zipFile, byte *zipFile_end, byte *targetFolder, byte *targetFolder_end, byte *password);
        byte[] __bytes_zipFile = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(zipFile.Length)];
        int __len_zipFile = System.Text.Encoding.UTF8.GetBytes(zipFile, __bytes_zipFile);
        fixed (byte *__ptr_zipFile = __bytes_zipFile)
        {
            byte[] __bytes_targetFolder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(targetFolder.Length)];
            int __len_targetFolder = System.Text.Encoding.UTF8.GetBytes(targetFolder, __bytes_targetFolder);
            fixed (byte *__ptr_targetFolder = __bytes_targetFolder)
            {
                byte __deref_password = password.GetValueOrDefault();
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_decompressZip_std_filesystem_path(__ptr_zipFile, __ptr_zipFile + __len_zipFile, __ptr_targetFolder, __ptr_targetFolder + __len_targetFolder, password.HasValue ? &__deref_password : null), is_owning: true));
            }
        }
    }

    /**
    * \brief decompresses given binary stream (containing the data of a zip file only) into given folder
    * \param password if password is given then it will be used to decipher encrypted archive
    */
    /// Generated from function `MR::decompressZip`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> DecompressZip(MR.Std.Istream zipStream, ReadOnlySpan<char> targetFolder, byte? password = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decompressZip_std_istream", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_decompressZip_std_istream(MR.Std.Istream._Underlying *zipStream, byte *targetFolder, byte *targetFolder_end, byte *password);
        byte[] __bytes_targetFolder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(targetFolder.Length)];
        int __len_targetFolder = System.Text.Encoding.UTF8.GetBytes(targetFolder, __bytes_targetFolder);
        fixed (byte *__ptr_targetFolder = __bytes_targetFolder)
        {
            byte __deref_password = password.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_decompressZip_std_istream(zipStream._UnderlyingPtr, __ptr_targetFolder, __ptr_targetFolder + __len_targetFolder, password.HasValue ? &__deref_password : null), is_owning: true));
        }
    }

    /**
    * \brief compresses given folder in given zip-file
    * \param excludeFiles files that should not be included to result zip 
    * \param password if password is given then the archive will be encrypted
    * \param cb an option to get progress notifications and cancel the operation
    */
    /// Generated from function `MR::compressZip`.
    /// Parameter `excludeFiles` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CompressZip(ReadOnlySpan<char> zipFile, ReadOnlySpan<char> sourceFolder, MR.Std.Const_Vector_StdFilesystemPath? excludeFiles = null, byte? password = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compressZip", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_compressZip(byte *zipFile, byte *zipFile_end, byte *sourceFolder, byte *sourceFolder_end, MR.Std.Const_Vector_StdFilesystemPath._Underlying *excludeFiles, byte *password, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        byte[] __bytes_zipFile = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(zipFile.Length)];
        int __len_zipFile = System.Text.Encoding.UTF8.GetBytes(zipFile, __bytes_zipFile);
        fixed (byte *__ptr_zipFile = __bytes_zipFile)
        {
            byte[] __bytes_sourceFolder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(sourceFolder.Length)];
            int __len_sourceFolder = System.Text.Encoding.UTF8.GetBytes(sourceFolder, __bytes_sourceFolder);
            fixed (byte *__ptr_sourceFolder = __bytes_sourceFolder)
            {
                byte __deref_password = password.GetValueOrDefault();
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_compressZip(__ptr_zipFile, __ptr_zipFile + __len_zipFile, __ptr_sourceFolder, __ptr_sourceFolder + __len_sourceFolder, excludeFiles is not null ? excludeFiles._UnderlyingPtr : null, password.HasValue ? &__deref_password : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}

public static partial class MR
{
    /// how to determine the sign of distances from a mesh
    public enum SignDetectionMode : int
    {
        /// unsigned distance, useful for bidirectional `Shell` offset
        Unsigned = 0,
        /// sign detection from OpenVDB library, which is good and fast if input geometry is closed
        OpenVDB = 1,
        /// the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
        ProjectionNormal = 2,
        /// ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh;
        /// this mode is slow, and it does NOT have CUDA acceleration at this moment
        WindingRule = 3,
        /// computes robust winding number generalization with support of holes and self-intersections in mesh,
        /// it is the slowest sign detection mode, but it CAN be accelerated with CUDA if this mode activated e.g. in OffsetParameters.fwn
        HoleWindingRule = 4,
    }

    /// how to determine the sign of distances from a mesh, short version including auto-detection
    public enum SignDetectionModeShort : int
    {
        ///< automatic selection of the fastest method among safe options for the current mesh
        Auto = 0,
        ///< detects sign from the winding number generalization with support for holes and self-intersections in mesh
        HoleWindingNumber = 1,
        ///< detects sign from the pseudonormal in closest mesh point, which is fast but unsafe in the presence of holes and self-intersections in mesh
        ProjectionNormal = 2,
    }

    /// returns string representation of enum values
    /// Generated from function `MR::asString`.
    public static unsafe byte? AsString(MR.SignDetectionMode m)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_asString_MR_SignDetectionMode", ExactSpelling = true)]
        extern static byte *__MR_asString_MR_SignDetectionMode(MR.SignDetectionMode m);
        var __ret = __MR_asString_MR_SignDetectionMode(m);
        return __ret is not null ? *__ret : null;
    }
}

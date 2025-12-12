public static partial class MR
{
    public static partial class PointCloudComponents
    {
        /// returns the union of point cloud components containing at least minSize points and connected by a distance no greater than \param maxDist
        /// \param minSize must be more than 1
        /// Generated from function `MR::PointCloudComponents::getLargeComponentsUnion`.
        /// Parameter `pc` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> GetLargeComponentsUnion(MR.Const_PointCloud pointCloud, float maxDist, int minSize, MR.Std._ByValue_Function_BoolFuncFromFloat? pc = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudComponents_getLargeComponentsUnion_MR_PointCloud", ExactSpelling = true)]
            extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_PointCloudComponents_getLargeComponentsUnion_MR_PointCloud(MR.Const_PointCloud._Underlying *pointCloud, float maxDist, int minSize, MR.Misc._PassBy pc_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *pc);
            return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_PointCloudComponents_getLargeComponentsUnion_MR_PointCloud(pointCloud._UnderlyingPtr, maxDist, minSize, pc is not null ? pc.PassByMode : MR.Misc._PassBy.default_arg, pc is not null && pc.Value is not null ? pc.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the union of vertices components containing at least minSize points
        /// \param unionStructs prepared point union structure
        /// \note have side effect: call unionStructs.roots() that change unionStructs
        /// Generated from function `MR::PointCloudComponents::getLargeComponentsUnion`.
        /// Parameter `pc` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> GetLargeComponentsUnion(MR.UnionFind_MRVertId unionStructs, MR.Const_VertBitSet region, int minSize, MR.Std._ByValue_Function_BoolFuncFromFloat? pc = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudComponents_getLargeComponentsUnion_MR_UnionFind_MR_VertId", ExactSpelling = true)]
            extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_PointCloudComponents_getLargeComponentsUnion_MR_UnionFind_MR_VertId(MR.UnionFind_MRVertId._Underlying *unionStructs, MR.Const_VertBitSet._Underlying *region, int minSize, MR.Misc._PassBy pc_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *pc);
            return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_PointCloudComponents_getLargeComponentsUnion_MR_UnionFind_MR_VertId(unionStructs._UnderlyingPtr, region._UnderlyingPtr, minSize, pc is not null ? pc.PassByMode : MR.Misc._PassBy.default_arg, pc is not null && pc.Value is not null ? pc.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// returns vector of point cloud components containing at least minSize points and connected by a distance no greater than \param maxDist
        /// \param minSize must be more than 1
        /// Generated from function `MR::PointCloudComponents::getLargeComponents`.
        /// Parameter `pc` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRVertBitSet_StdString> GetLargeComponents(MR.Const_PointCloud pointCloud, float maxDist, int minSize, MR.Std._ByValue_Function_BoolFuncFromFloat? pc = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudComponents_getLargeComponents", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRVertBitSet_StdString._Underlying *__MR_PointCloudComponents_getLargeComponents(MR.Const_PointCloud._Underlying *pointCloud, float maxDist, int minSize, MR.Misc._PassBy pc_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *pc);
            return MR.Misc.Move(new MR.Expected_StdVectorMRVertBitSet_StdString(__MR_PointCloudComponents_getLargeComponents(pointCloud._UnderlyingPtr, maxDist, minSize, pc is not null ? pc.PassByMode : MR.Misc._PassBy.default_arg, pc is not null && pc.Value is not null ? pc.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// gets all components of point cloud connected by a distance no greater than \param maxDist
        /// \detail if components number more than the maxComponentCount, they will be combined into groups of the same size 
        /// \note be careful, if point cloud is large enough and has many components, the memory overflow will occur
        /// \param maxComponentCount should be more then 1
        /// \return pair components bitsets vector and number components in one group if components number more than maxComponentCount
        /// Generated from function `MR::PointCloudComponents::getAllComponents`.
        /// Parameter `maxComponentCount` defaults to `2147483647`.
        /// Parameter `pc` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdPairStdVectorMRVertBitSetInt_StdString> GetAllComponents(MR.Const_PointCloud pointCloud, float maxDist, int? maxComponentCount = null, MR.Std._ByValue_Function_BoolFuncFromFloat? pc = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudComponents_getAllComponents", ExactSpelling = true)]
            extern static MR.Expected_StdPairStdVectorMRVertBitSetInt_StdString._Underlying *__MR_PointCloudComponents_getAllComponents(MR.Const_PointCloud._Underlying *pointCloud, float maxDist, int *maxComponentCount, MR.Misc._PassBy pc_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *pc);
            int __deref_maxComponentCount = maxComponentCount.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_StdPairStdVectorMRVertBitSetInt_StdString(__MR_PointCloudComponents_getAllComponents(pointCloud._UnderlyingPtr, maxDist, maxComponentCount.HasValue ? &__deref_maxComponentCount : null, pc is not null ? pc.PassByMode : MR.Misc._PassBy.default_arg, pc is not null && pc.Value is not null ? pc.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// gets union - find structure for vertices in \param region connected by a distance no greater than \param maxDist 
        /// Generated from function `MR::PointCloudComponents::getUnionFindStructureVerts`.
        /// Parameter `pc` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRUnionFindMRVertId_StdString> GetUnionFindStructureVerts(MR.Const_PointCloud pointCloud, float maxDist, MR.Const_VertBitSet? region = null, MR.Std._ByValue_Function_BoolFuncFromFloat? pc = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudComponents_getUnionFindStructureVerts", ExactSpelling = true)]
            extern static MR.Expected_MRUnionFindMRVertId_StdString._Underlying *__MR_PointCloudComponents_getUnionFindStructureVerts(MR.Const_PointCloud._Underlying *pointCloud, float maxDist, MR.Const_VertBitSet._Underlying *region, MR.Misc._PassBy pc_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *pc);
            return MR.Misc.Move(new MR.Expected_MRUnionFindMRVertId_StdString(__MR_PointCloudComponents_getUnionFindStructureVerts(pointCloud._UnderlyingPtr, maxDist, region is not null ? region._UnderlyingPtr : null, pc is not null ? pc.PassByMode : MR.Misc._PassBy.default_arg, pc is not null && pc.Value is not null ? pc.Value._UnderlyingPtr : null), is_owning: true));
        }
    }
}

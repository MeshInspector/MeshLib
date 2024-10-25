using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using static MR.DotNet.Vector3f;

namespace MR.DotNet
{
    using VertId = int;    
    using FaceId = int;
    using VertMap = List<int>;
    using VertMapReadOnly = ReadOnlyCollection<int>;

    using FaceMap = List<int>;
    using FaceMapReadOnly = ReadOnlyCollection<int>;

    public enum MapObject
    {
        A = 0,
        B = 1,
        Count = 2
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRFaceMap
    {
        public IntPtr data;
        public ulong size;
        public IntPtr reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRVertMap
    {
        public IntPtr data;
        public ulong size;
        public IntPtr reserved;
    };

    public class BooleanMaps
    {

        /// "after cut" faces to "origin" faces
        /// this map is not 1-1, but N-1
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFaceMap mrBooleanResultMapperMapsCut2origin(IntPtr maps);

        /// "after cut" faces to "after stitch" faces (1-1)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFaceMap mrBooleanResultMapperMapsCut2newFaces(IntPtr maps);

        /// "origin" vertices to "after stitch" vertices (1-1)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRVertMap mrBooleanResultMapperMapsOld2NewVerts(IntPtr maps);

        /// old topology indexes are valid if true
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern bool mrBooleanResultMapperMapsIdentity(IntPtr maps);

        internal BooleanMaps(IntPtr maps)
        {
            maps_ = maps;
        }

        public FaceMapReadOnly Cut2Origin
        {
            get
            {
                if ( cut2origin_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsCut2origin(maps_);
                    cut2origin_ = new List<FaceId>( (int)mrMap.size );
                    int sizeOfFaceId= Marshal.SizeOf(typeof(MRFaceId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentFacePtr = IntPtr.Add(mrMap.data, i * sizeOfFaceId);
                        var face = Marshal.PtrToStructure<MRFaceId>(currentFacePtr);
                        cut2origin_.Add(face.id);
                    }
                }

                return cut2origin_.AsReadOnly();
            }
        }

        public FaceMapReadOnly Cut2NewFaces
        {
            get
            {
                if ( cut2newFaces_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsCut2newFaces(maps_);
                    cut2newFaces_ = new List<FaceId>( (int)mrMap.size );
                    int sizeOfFaceId= Marshal.SizeOf(typeof(MRFaceId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentFacePtr = IntPtr.Add(mrMap.data, i * sizeOfFaceId);
                        var face = Marshal.PtrToStructure<MRFaceId>(currentFacePtr);
                        cut2newFaces_.Add(face.id);
                    }
                }

                return cut2newFaces_.AsReadOnly();
            }
        }

        public VertMapReadOnly Old2NewVerts
        {
            get
            {
                if ( old2newVerts_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsOld2NewVerts(maps_);
                    old2newVerts_ = new List<VertId>( (int)mrMap.size );
                    int sizeOfVertId= Marshal.SizeOf(typeof(MRVertId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentVertPtr = IntPtr.Add(mrMap.data, i * sizeOfVertId);
                        var vert = Marshal.PtrToStructure<MRVertId>(currentVertPtr);
                        old2newVerts_.Add(vert.id);
                    }
                }

                return old2newVerts_.AsReadOnly();
            }
        }

        public bool Identity
        {
            get
            {
                return mrBooleanResultMapperMapsIdentity(maps_);
            }
        }

        IntPtr maps_;
        FaceMap? cut2origin_;
        FaceMap? cut2newFaces_;
        VertMap? old2newVerts_;
    }
    public class BooleanResultMapper : IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperNew();

        /// Returns faces bitset of result mesh corresponding input one
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapFaces(IntPtr mapper, IntPtr oldBS, MapObject obj);

        /// Returns vertices bitset of result mesh corresponding input one
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapVerts(IntPtr mapper, IntPtr oldBS, MapObject obj);


        /// Returns only new faces that are created during boolean operation
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperNewFaces(IntPtr mapper);

        /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperFilteredOldFaceBitSet(IntPtr mapper, IntPtr oldBS, MapObject obj);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperGetMaps(IntPtr mapper, MapObject index);


        /// deallocates a BooleanResultMapper object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrBooleanResultMapperFree(IntPtr mapper);

        #region constructor and destructor

        public BooleanResultMapper()
        {
            mapper_ = mrBooleanResultMapperNew();
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mapper_ != IntPtr.Zero)
                {
                    mrBooleanResultMapperFree(mapper_);
                }

                disposed = true;
            }
        }

        ~BooleanResultMapper()
        {
            Dispose(false);
        }
        #endregion
        #region properties

        public BitSet FaceMap( BitSet oldBS, MapObject obj)
        {
            if ( maps_ is null )
                maps_ = new BooleanMaps?[2];

            if ( maps_[(int)obj] is null )
            {
                maps_[(int)obj] = new BooleanMaps( mrBooleanResultMapperGetMaps(mapper_, obj) );
            }

            return new BitSet(mrBooleanResultMapperMapFaces(mapper_, oldBS.bs_, obj));
        }

        public BitSet VertMap( BitSet oldBS, MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            if (maps_[(int)obj] is null)
            {
                maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return new BitSet(mrBooleanResultMapperMapVerts(mapper_, oldBS.bs_, obj));
        }

        public BitSet NewFaces()
        {
            return new BitSet(mrBooleanResultMapperNewFaces(mapper_));
        }

        public BooleanMaps GetMaps( MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            if (maps_[(int)obj] is null)
            {
                maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return maps_[(int)obj];
        }

        public BitSet FilteredOldFaceBitSet( BitSet oldBS, MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            if (maps_[(int)obj] is null)
            {
                maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return new BitSet(mrBooleanResultMapperFilteredOldFaceBitSet(mapper_, oldBS.bs_, obj));
        }

        internal IntPtr Mapper { get { return mapper_; } }
        #endregion
        #region private fields

        IntPtr mapper_;
        BooleanMaps?[]? maps_;
        #endregion
    }
}

﻿using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;

namespace MR
{

    public partial class DotNet
    {
        public class SelfIntersections
        {
            public enum Method
            {
                Relax,
                CutAndFill
            };

            public struct Settings
            {
                /// If true then count touching faces as self-intersections
                public bool touchIsIntersection = true;
                /// Fix method
                public Method method = Method.Relax;
                /// Maximum relax iterations
                public int relaxIterations = 5;
                /// Maximum expand count (edge steps from self-intersecting faces), should be > 0
                public int maxExpand = 3;
                /// Edge length for subdivision of holes covers (0.0f means auto)
                /// FLT_MAX to disable subdivision
                public float subdivideEdgeLen = 0;
                public Settings() { }
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct MRFixSelfIntersectionsSettings
            {
                public byte touchIsIntersection = 1;
                public Method method = Method.Relax;
                public int relaxIterations = 5;
                public int maxExpand = 3;
                public float subdivideEdgeLen = 0;
                public IntPtr cb = IntPtr.Zero;
                public MRFixSelfIntersectionsSettings() { }
            };

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrStringData(IntPtr str);


            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrFixSelfIntersectionsGetFaces(IntPtr mesh, bool touchIsIntersection, IntPtr cb, ref IntPtr errorString);


            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrFixSelfIntersectionsFix(IntPtr mesh, ref MRFixSelfIntersectionsSettings settings, ref IntPtr errorString);

            /// Find all self-intersections faces component-wise
            static public FaceBitSet GetFaces(Mesh mesh,bool touchIsIntersection = true)
            {
                IntPtr errorStr = IntPtr.Zero;
                var mrFaces = mrFixSelfIntersectionsGetFaces(mesh.mesh_, touchIsIntersection, IntPtr.Zero, ref errorStr);

                if (errorStr != IntPtr.Zero)
                {
                    var errData = mrStringData(errorStr);
                    string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                    throw new SystemException(errorMessage);
                }

                return new FaceBitSet(mrFaces);
            }
            /// Finds and fixes self-intersections per component:
            static public void Fix(ref Mesh mesh, Settings settings)
            {
                IntPtr errorStr = IntPtr.Zero;

                MRFixSelfIntersectionsSettings mrSettings = new MRFixSelfIntersectionsSettings();
                mrSettings.touchIsIntersection = settings.touchIsIntersection ? (byte)1 : (byte)0;
                mrSettings.method = settings.method;
                mrSettings.relaxIterations = settings.relaxIterations;
                mrSettings.maxExpand = settings.maxExpand;
                mrSettings.subdivideEdgeLen = settings.subdivideEdgeLen;                

                mrFixSelfIntersectionsFix(mesh.mesh_, ref mrSettings, ref errorStr);
                if (errorStr != IntPtr.Zero)
                {
                    var errData = mrStringData(errorStr);
                    string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                    throw new SystemException(errorMessage);
                }
            }
        }
    }
}

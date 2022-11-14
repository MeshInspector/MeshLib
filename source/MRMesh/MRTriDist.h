#pragma once

#include "MRVector.h"

/*************************************************************************\

  Copyright 1999 The University of North Carolina at Chapel Hill.
  All Rights Reserved.

  Permission to use, copy, modify and distribute this software and its
  documentation for educational, research and non-profit purposes, without
  fee, and without a written agreement is hereby granted, provided that the
  above copyright notice and the following three paragraphs appear in all
  copies.

  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
  LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
  CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
  USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
  OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGES.

  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
  PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
  NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

  The authors may be contacted via:

  US Mail:             E. Larsen
                       Department of Computer Science
                       Sitterson Hall, CB #3175
                       University of N. Carolina
                       Chapel Hill, NC 27599-3175

  Phone:               (919)962-1749

  EMail:               geom@cs.unc.edu


\**************************************************************************/

namespace MR
{

/// \brief computes the closest points on two triangles, and returns the 
/// squared distance between them.
/// 
/// \param s,t are the triangles, stored tri[point][dimension].
///
/// \details If the triangles are disjoint, p and q give the closest points of 
/// s and t respectively. However, if the triangles overlap, p and q 
/// are basically a random pair of points from the triangles, not 
/// coincident points on the intersection of the triangles, as might 
/// be expected.
MRMESH_API float TriDist( Vector3f & p, Vector3f & q, const Vector3f s[3], const Vector3f t[3] );

/// Returns closest points between an segment pair.
MRMESH_API void SegPoints(
          // if both closest points are in segment endpoints, then directed from closest point 1 to closest point 2,
          // if both closest points are inner to the segments, then its orthogonal to both segments and directed from 1 to 2,
          // otherwise it is orthogonal to the segment with inner closest point and rotated toward/away the other closest point in endpoint
          Vector3f & VEC,
          Vector3f & X, Vector3f & Y,             // closest points
          const Vector3f & P, const Vector3f & A, // seg 1 origin, vector
          const Vector3f & Q, const Vector3f & B);// seg 2 origin, vector

} // namespace MR

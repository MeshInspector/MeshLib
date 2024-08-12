#include "MRTriDist.h"

namespace MR
{

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

//--------------------------------------------------------------------------
// File:   TriDist.cpp
// Author: Eric Larsen
// Description:
// contains SegPoints() for finding closest points on a pair of line
// segments and TriDist() for finding closest points on a pair of triangles
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
// SegPoints() 
//
// Returns closest points between an segment pair.
// Implemented from an algorithm described in
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.   
//--------------------------------------------------------------------------

void
SegPoints( Vector3f & VEC, 
          Vector3f & X, Vector3f & Y,             // closest points
          const Vector3f & P, const Vector3f & A, // seg 1 origin, vector
          const Vector3f & Q, const Vector3f & B) // seg 2 origin, vector
{
  Vector3f T, TMP;
  float A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;

  T = Q - P;
  A_dot_A = dot(A,A);
  B_dot_B = dot(B,B);
  A_dot_B = dot(A,B);
  A_dot_T = dot(A,T);
  B_dot_T = dot(B,T);

  // t parameterizes ray P,A 
  // u parameterizes ray Q,B 

  float t,u;

  // compute t for the closest point on ray P,A to
  // ray Q,B

  float denom = A_dot_A*B_dot_B - A_dot_B*A_dot_B;

  t = (A_dot_T*B_dot_B - B_dot_T*A_dot_B) / denom;

  // clamp result so t is on the segment P,A

  if ((t < 0) || std::isnan(t)) t = 0; else if (t > 1) t = 1;

  // find u for point on ray Q,B closest to point at t

  u = (t*A_dot_B - B_dot_T) / B_dot_B;

  // if u is on segment Q,B, t and u correspond to 
  // closest points, otherwise, clamp u, recompute and
  // clamp t 

  if ((u <= 0) || std::isnan(u)) {

    Y = Q;

    t = A_dot_T / A_dot_A;

    if ((t <= 0) || std::isnan(t)) {
      X = P;
      VEC = Q - P;
    }
    else if (t >= 1) {
      X = P + A;
      VEC = Q - X;
    }
    else {
      X = P +  A * t;
      TMP = cross( T, A );
      VEC = cross( A, TMP );
    }
  }
  else if (u >= 1) {

    Y =  Q + B;

    t = (A_dot_B + A_dot_T) / A_dot_A;

    if ((t <= 0) || std::isnan(t)) {
      X = P;
      VEC = Y - P;
    }
    else if (t >= 1) {
      X = P + A;
      VEC = Y - X;
    }
    else {
      X = P + A * t;
      T = Y - P;
      TMP = cross( T, A );
      VEC = cross( A, TMP );
    }
  }
  else {

    Y = Q + B * u;

    if ((t <= 0) || std::isnan(t)) {
      X = P;
      TMP = cross( T, B );
      VEC = cross( B, TMP );
    }
    else if (t >= 1) {
      X = P + A;
      T = Q - X;
      TMP = cross( T, B );
      VEC = cross( B, TMP );
    }
    else {
      X = P + A * t;
      VEC = cross( A, B );
      if (dot(VEC, T) < 0) {
        VEC = -VEC;
      }
    }
  }
}

//--------------------------------------------------------------------------
// TriDist() 
//
// Computes the closest points on two triangles, and returns the 
// squared distance between them.
// 
// S and T are the triangles, stored tri[point][dimension].
//
// If the triangles are disjoint, P and Q give the closest points of 
// S and T respectively. However, if the triangles overlap, P and Q 
// are basically a random pair of points from the triangles, not 
// coincident points on the intersection of the triangles, as might 
// be expected.
//--------------------------------------------------------------------------

float TriDist( Vector3f & P, Vector3f & Q, const Vector3f S[3], const Vector3f T[3] )
{
  // Compute vectors along the 6 sides

  Vector3f Sv[3], Tv[3];
  Vector3f VEC;

  Sv[0] = S[1] - S[0];
  Sv[1] = S[2] - S[1];
  Sv[2] = S[0] - S[2];

  Tv[0] = T[1] - T[0];
  Tv[1] = T[2] - T[1];
  Tv[2] = T[0] - T[2];

  // For each edge pair, the vector connecting the closest points 
  // of the edges defines a slab (parallel planes at head and tail
  // enclose the slab). If we can show that the off-edge vertex of 
  // each triangle is outside of the slab, then the closest points
  // of the edges are the closest points for the triangles.
  // Even if these tests fail, it may be helpful to know the closest
  // points found, and whether the triangles were shown disjoint

  Vector3f V, Z, minP, minQ;
  float mindd;
  int shown_disjoint = 0;

  mindd = (S[0] - T[0]).lengthSq() + 1;  // Set first minimum safely high

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      // Find closest points on edges i & j, plus the 
      // vector (and distance squared) between these points

      SegPoints(VEC,P,Q,S[i],Sv[i],T[j],Tv[j]);
      
      V = Q - P;
      float dd = dot(V,V);

      // Verify this closest point pair only if the distance 
      // squared is less than the minimum found thus far.

      if (dd <= mindd)
      {
        minP = P;
        minQ = Q;
        mindd = dd;

        Z = S[(i+2)%3] - P;
        float a = dot(Z,VEC);
        Z = T[(j+2)%3] - Q;
        float b = dot(Z,VEC);

        if ((a <= 0) && (b >= 0)) 
            return dd;

        float p = dot(V, VEC);

        if (a < 0) a = 0;
        if (b > 0) b = 0;
        if ((p - a + b) > 0) shown_disjoint = 1;	
      }
    }
  }

  // No edge pairs contained the closest points.  
  // either:
  // 1. one of the closest points is a vertex, and the
  //    other point is interior to a face.
  // 2. the triangles are overlapping.
  // 3. an edge of one triangle is parallel to the other's face. If
  //    cases 1 and 2 are not true, then the closest points from the 9
  //    edge pairs checks above can be taken as closest points for the
  //    triangles.
  // 4. possibly, the triangles were degenerate.  When the 
  //    triangle points are nearly colinear or coincident, one 
  //    of above tests might fail even though the edges tested
  //    contain the closest points.

  // First check for case 1

  Vector3f Sn = cross( Sv[0], Sv[1] ); // Compute normal to S triangle
  float Snl = dot(Sn,Sn);      // Compute square of length of normal
  
  // If cross product is long enough,

  if (Snl > 1e-15)  
  {
    // Get projection lengths of T points

    float Tp[3]; 

    V = S[0] - T[0];
    Tp[0] = dot(V,Sn);

    V = S[0] - T[1];
    Tp[1] = dot(V,Sn);

    V = S[0] - T[2];
    Tp[2] = dot(V,Sn);

    // If Sn is a separating direction,
    // find point with smallest projection

    int point = -1;
    if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
    {
      if (Tp[0] < Tp[1]) point = 0; else point = 1;
      if (Tp[2] < Tp[point]) point = 2;
    }
    else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
    {
      if (Tp[0] > Tp[1]) point = 0; else point = 1;
      if (Tp[2] > Tp[point]) point = 2;
    }

    // If Sn is a separating direction, 

    if (point >= 0) 
    {
      shown_disjoint = 1;

      // Test whether the point found, when projected onto the 
      // other triangle, lies within the face.
    
      V = T[point] - S[0];
      Z = cross( Sn, Sv[0] );
      if (dot(V,Z) > 0)
      {
        V = T[point] - S[1];
        Z = cross( Sn, Sv[1] );
        if (dot(V,Z) > 0)
        {
          V = T[point] - S[2];
          Z = cross( Sn, Sv[2] );
          if (dot(V,Z) > 0)
          {
            // T[point] passed the test - it's a closest point for 
            // the T triangle; the other point is on the face of S

            P = T[point] + Sn * Tp[point]/Snl;
            Q = T[point];
            return ( P - Q ).lengthSq();
          }
        }
      }
    }
  }

  Vector3f Tn = cross( Tv[0], Tv[1] ); 
  float Tnl = dot(Tn,Tn);      
  
  if (Tnl > 1e-15)  
  {
    float Sp[3]; 

    V = T[0] - S[0];
    Sp[0] = dot(V,Tn);

    V = T[0] - S[1];
    Sp[1] = dot(V,Tn);

    V = T[0] - S[2];
    Sp[2] = dot(V,Tn);

    int point = -1;
    if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
    {
      if (Sp[0] < Sp[1]) point = 0; else point = 1;
      if (Sp[2] < Sp[point]) point = 2;
    }
    else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
    {
      if (Sp[0] > Sp[1]) point = 0; else point = 1;
      if (Sp[2] > Sp[point]) point = 2;
    }

    if (point >= 0) 
    { 
      shown_disjoint = 1;

      V = S[point] - T[0];
      Z = cross( Tn, Tv[0] );
      if (dot(V,Z) > 0)
      {
        V = S[point] - T[1];
        Z = cross( Tn, Tv[1] );
        if (dot(V,Z) > 0)
        {
          V = S[point] - T[2];
          Z = cross( Tn, Tv[2] );
          if (dot(V,Z) > 0)
          {
            P = S[point];
            Q = S[point] + Tn * Sp[point]/Tnl;
            return ( P - Q ).lengthSq();
          }
        }
      }
    }
  }

  // Case 1 can't be shown.
  // If one of these tests showed the triangles disjoint,
  // we assume case 3 or 4, otherwise we conclude case 2, 
  // that the triangles overlap.
  
  if (shown_disjoint)
  {
    P = minP;
    Q = minQ;
    return mindd;
  }

  P = Q = 0.5f * (P + Q);
  return 0;
}

} // namespace MR

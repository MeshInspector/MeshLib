#pragma once

// this is to include all important for us Eigen parts in a precompiled header and suppress warnings there

#pragma warning(push)
#pragma warning(disable:4464)  //relative include path contains '..'
#pragma warning(disable:5054)  //operator '&': deprecated between enumerations of different types
#pragma warning(disable:6011)  //Dereferencing NULL pointer 'newOuterIndex'. 
#pragma warning(disable:6255)  //_alloca indicates failure by raising a stack overflow exception.  Consider using _malloca instead.
#pragma warning(disable:6294)  //Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.
#pragma warning(disable:6385)  //Reading invalid data from 'newOuterIndex':  the readable size is '(Index)(m_outerSize+1)*sizeof(StorageIndex)' bytes, but '8' bytes may be read.
#pragma warning(disable:6386)  //Buffer overrun while writing to 'newOuterIndex':  the writable size is '(Index)(m_outerSize+1)*sizeof(StorageIndex)' bytes, but 'm_outerSize' bytes might be written.
#pragma warning(disable:6387)  //'m_outerIndex' could be '0':  this does not adhere to the specification for the function 'memset'. 
#pragma warning(disable:26450) //Arithmetic overflow: '<<' operation causes overflow at compile time. Use a wider type to store the operands (io.1).
#pragma warning(disable:26451) //Arithmetic overflow: Using operator '-' on a 4 byte value and then casting the result to a 8 byte value. Cast the value to the wider type before calling operator '-' to avoid overflow (io.2).
#pragma warning(disable:26454) //Arithmetic overflow: '-' operation produces a negative unsigned result at compile time (io.5).
#pragma warning(disable:26495) //Variable 'Eigen::internal::Packet1cd::v' is uninitialized. Always initialize a member variable (type.6).
#pragma warning(disable:26812) //The enum type 'Eigen::Action' is unscoped. Prefer 'enum class' over 'enum' (Enum.3).
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
#pragma warning(pop)

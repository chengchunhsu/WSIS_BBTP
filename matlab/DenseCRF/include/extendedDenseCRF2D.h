#pragma once

#include <string>     
#include <stdexcept>  
#include "densecrf/include/densecrf.h"

// Missing in densecrf.h
void expAndNormalize ( Eigen::MatrixXf & out, const Eigen::MatrixXf & in );

class extendedDenseCRF2D : public DenseCRF2D
{
public:
  extendedDenseCRF2D( int W, int H, int M) : DenseCRF2D(W,H,M)
  {}

  double energy(VectorXs map)
  {
    Eigen::VectorXf unary = unaryEnergy( map );
    Eigen::VectorXf pairwise = pairwiseEnergy( map );

    double energy = 0;

    for (int i = 0; i < W_*H_; i++)
      energy += unary(i);

    for (int i = 0; i < W_*H_; i++)
      energy -= pairwise(i);

    // Correct for negative weighted potts.
    // the pairwise cost is -w[a==b], but the paper states w[a!=b].
    map *= 0;
    pairwise = pairwiseEnergy( map );
    
    for (int i = 0; i < W_*H_; i++)
      energy += pairwise(i);

    return energy;
  }
};
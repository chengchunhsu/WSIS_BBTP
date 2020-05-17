#include "meanfield.h"
#include "solvers.h"
double timer;

void mexFunction(int nlhs, 		    /* number of expected outputs */
        mxArray        *plhs[],	    /* mxArray output pointer array */
        int            nrhs, 		/* number of inputs */
        const mxArray  *prhs[]		/* mxArray input pointer array */)
{
  const matrix<unsigned char> im_matrix(prhs[0]);
  const matrix<float>  unary_matrix(prhs[1]);
  const matrix<unsigned int> im_size(prhs[2]);
  matrix<signed short> segmentation(prhs[3]);

  const unsigned char * image  = im_matrix.data;
  float * unary_array      = unary_matrix.data;
  short * map = segmentation.data;

  //Structure to hold and parse additional parameters
  MexParams params(1, prhs+4);

  // Weights used to define the energy function
  PairwiseWeights pairwiseWeights(params);

  KernelType kerneltype = CONST_KERNEL;
  NormalizationType normalizationtype = parseNormalizationType(params);

  // The image
  const int M = im_size(0);
  const int N = im_size(1);
  const int C = im_size(2);

  // Calculate number of labels
  const int UC = unary_matrix.numel();
  const int numberOfLabels = UC/(M*N);
  int numVariables = M*N;

  // Creating linear index functor
  LinearIndex unaryLinearIndex(M,N, numberOfLabels);
  LinearIndex imageLinearIndex(M,N, C);

  const bool debug = params.get<bool>("debug", false);
  const bool calculate_exact_energy = params.get<bool>("calculate_exact_energy", false);

  extendedDenseCRF2D crf(M,N,numberOfLabels);
  Eigen::Map<Eigen::MatrixXf> unary(unary_array, numberOfLabels, numVariables);

  crf.setUnaryEnergy( unary );


  // Setup  pairwise cost
  crf.addPairwiseGaussian(pairwiseWeights.gaussian_x_stddev, 
                          pairwiseWeights.gaussian_y_stddev, 
                          new PottsCompatibility(pairwiseWeights.gaussian_weight),
                          kerneltype,
                          normalizationtype);

  crf.addPairwiseBilateral(pairwiseWeights.bilateral_x_stddev, 
                           pairwiseWeights.bilateral_y_stddev,
                           pairwiseWeights.bilateral_r_stddev, 
                           pairwiseWeights.bilateral_g_stddev, 
                           pairwiseWeights.bilateral_b_stddev,
                           image,
                           new PottsCompatibility(pairwiseWeights.bilateral_weight),
                           kerneltype,
                           normalizationtype);
          

  if (debug)
    std::cout << pairwiseWeights << std::endl;
  
  matrix<double> energy(1);

  //The following code calculates the correct energy but is very slow
  if (calculate_exact_energy)
  {
    UnaryCost unaryCost(unary_array, unaryLinearIndex);
    PairwiseCost pairwiseCost(image, pairwiseWeights, imageLinearIndex);
    EnergyFunctor energyFunctor(unaryCost, pairwiseCost, M,N, numberOfLabels);
    energy(0) = energyFunctor(map);
  } else
  {
    energy(0) = std::numeric_limits<double>::quiet_NaN();
  }

  matrix<double> mf_energy(1);
  Eigen::Map<VectorXs> map_eigen(map, numVariables);
  mf_energy(0) = crf.energy(map_eigen);
    
  plhs[0] = energy;
  plhs[1] = mf_energy;
}
// Decompress probabilities
#include "mex.h"
#include "mexutils.h"
#include "cppmatrix.h"
#include "probImage/probimage.h"

void mexFunction(int nlhs, 		    /* number of expected outputs */
        mxArray        *plhs[],	    /* mxArray output pointer array */
        int            nrhs, 		/* number of inputs */
        const mxArray  *prhs[]		/* mxArray input pointer array */)
{
  char *path = mxArrayToString(prhs[0]);

  ProbImage im;
  im.decompress(path);

  const float * data = im.data();

  matrix<double> unary(im.height(), 
                       im.width(),
                       im.depth());


  for (int x = 0; x < im.width(); x++) {
  for (int y = 0; y < im.height(); y++) {
  for (int z = 0; z < im.depth(); z++) {
    unary(y,x,z) = im(x,y,z);
  }
  } 
  }

  plhs[0] = unary;
}
#pragma once
#include "mex.h"
#include "matrix.h"
#include "math.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string.h>
#include <limits>   
#include <utility>
#include <sstream> 
#include "mexutils.h"
#include "cppmatrix.h"
#include "pairwise.h"

// Keep track on time
extern double timer;

#ifdef USE_OPENMP
#include <omp.h>
double get_wtime()
{
  return ::omp_get_wtime();
}
#else
#include <ctime>
double get_wtime()
{
  return (double)std::time(0);
}
#endif

void startTime()
{
  timer = ::get_wtime();
}

double endTime()
{
  double current_time = ::get_wtime();
  double elapsed = current_time - timer;
  timer = current_time;

  return elapsed;
}

double endTime(const char* msg)
{
	double time = endTime();
  mexPrintf("Elapsed: %.04f. %s \n", time, msg);

  return time;
}
  

typedef std::vector<std::pair<int, int> > pixelpair;

// Weights needed to define energy functional
// Norm depends on the underlying data.
// Weight is a user given constant.
struct PairwiseWeights
{
  PairwiseWeights(MexParams params)
  {
    gaussian_x_stddev = (float)params.get<double>("gaussian_x_stddev", 3);
    gaussian_y_stddev = (float)params.get<double>("gaussian_y_stddev", 3);
    gaussian_weight = (float)params.get<double>("gaussian_weight", 1);
    
    bilateral_x_stddev = (float)params.get<double>("bilateral_x_stddev", 60);
    bilateral_y_stddev = (float)params.get<double>("bilateral_y_stddev", 60);
    bilateral_weight = (float)params.get<double>("bilateral_weight", 1);
    
    bilateral_r_stddev = (float)params.get<double>("bilateral_r_stddev", 20);
    bilateral_g_stddev = (float)params.get<double>("bilateral_g_stddev", 20);
    bilateral_b_stddev = (float)params.get<double>("bilateral_b_stddev", 20); 

    gaussian_norm = 1;
    bilateral_norm = 1;
  };

  float gaussian_x_stddev;
  float gaussian_y_stddev;
  float gaussian_weight;
  
  float bilateral_x_stddev;
  float bilateral_y_stddev;
  float bilateral_weight;
  
  float bilateral_r_stddev;
  float bilateral_g_stddev;
  float bilateral_b_stddev;

  // Normalizing factor
  float gaussian_norm;
  float bilateral_norm;
};

std::ostream& operator<<(std::ostream &out, const PairwiseWeights& p){
   out << "Parameters:" << std::endl
       << "Gaussian "  << std::endl
       << " Weight: " << p.gaussian_weight 
       << std::endl
       << " Std x: " << p.gaussian_x_stddev 
       << " Std y: " << p.gaussian_y_stddev
       << std::endl
       << " Normalization factor " << p.gaussian_norm
       << std::endl
       << "Bilateral" << std::endl
       << " Weight: " << p.bilateral_weight 
       << std::endl
       << " Std x: " << p.bilateral_x_stddev 
       << " Std y: " << p.bilateral_y_stddev 
       << std::endl
       << " Std r: " << p.bilateral_r_stddev
       << " Std g: " << p.bilateral_g_stddev
       << " Std b: " << p.bilateral_b_stddev 
       << std::endl
       << " Normalization factor " << p.bilateral_norm;

   return out;
}


NormalizationType parseNormalizationType(MexParams params)
{
  string normalizationtype_str = params.get<string>("NormalizationType");

  if (normalizationtype_str == "NO_NORMALIZATION")
    return NO_NORMALIZATION;
  else if (normalizationtype_str == "NORMALIZE_BEFORE")
    return NORMALIZE_BEFORE;
  else if (normalizationtype_str == "NORMALIZE_AFTER")
    return NORMALIZE_AFTER;
  else if (normalizationtype_str == "NORMALIZE_SYMMETRIC")
    return NORMALIZE_SYMMETRIC;
  else
    throw runtime_error("Unknown normalizationtype");
}

// unaryLinearIndex functor
// Note that the linear index is differs slightly from the normal MATLAB packing
// DenseCRF code uses
// x0y0c0 x0y0c1 ,  ....
class LinearIndex
{
  public:
    // Number of rows and number of classes
    LinearIndex (int _nx, int _ny, int _nc = 1) : nx(_nx), ny(_ny), nc(_nc)
    {};

    int operator() (int x, int y, int c)
    {
      return x*nc + y*nx*nc + c;
    }

    // 2D lookup
    int operator() (int x, int y)
    {
       ASSERT(nc == 1);
       return x*ny + y;
    }

    int numel()
    {
      return nx*ny*nc;
    }

  public:
    int nx;
    int ny;
    int nc;
};

class Linear2sub
{
public:
  Linear2sub (int _nx, int _ny) : nx(_nx), ny(_ny)
  {};

  std::pair<int,int> operator() (int i)
  {
    int y = i/nx;
    int x = i - y*nx;

    return std::make_pair(x,y);
  }

private: 
  const int nx,ny;
};

// Unary cost as functor
class UnaryCost
{
  public:
    UnaryCost(const float * _unary,  LinearIndex& _unaryLinearIndex) : 
      unary(_unary), unaryLinearIndex(_unaryLinearIndex)
    {};

    float operator() (std::pair<int,int>& p, int c)
    {
      return operator() (p.first, p.second, c);
    }

    float operator() (int x, int y, int c)
    {
      return unary[ unaryLinearIndex( x , y ,c)];
    }

  private:
      const float * unary;
      LinearIndex unaryLinearIndex;
};

inline double square(int x){
    return (double)x*x;
}

inline float square(float x){
  return x*x;
}

inline double square(double x){
  return x*x;
}

class PairwiseCost
{
  public:
    PairwiseCost(const unsigned char * _image, 
                 PairwiseWeights& weights,
                 LinearIndex& _imageLinearIndex) : 

                image(_image), 
                imageLinearIndex(_imageLinearIndex)
    { 
      wbx = 2* square(weights.bilateral_x_stddev);
      wby = 2* square(weights.bilateral_y_stddev);

      wbr = 2* square(weights.bilateral_r_stddev);
      wbg = 2* square(weights.bilateral_g_stddev);
      wbb = 2* square(weights.bilateral_b_stddev);

      wgx = 2* square(weights.gaussian_x_stddev);
      wgy = 2* square(weights.gaussian_y_stddev);

      bilateral_weight = weights.bilateral_weight;
      gaussian_weight = weights.gaussian_weight;
    };

  float operator() (std::pair<int, int>& p0, std::pair<int, int>& p1) 
  {
    return operator() (p0.first, p0.second, p1.first, p1.second);
  } 

  float operator()  (int x0, int y0, int x1, int y1)
  {

    float cost =  bilateral_weight*bilateral_cost(x0,y0,x1,y1);
          cost += gaussian_weight *gaussian_cost (x0,y0,x1,y1);


    return cost; 
  }

  float bilateral_cost(int x0, int y0, int x1, int y1)
  {
      return (float)exp(
              - ( square(x0-x1) / wbx ) 
              - ( square(y0-y1) / wby ) 
              - ( square(image[imageLinearIndex(x0,y0,0)] - image[imageLinearIndex(x1,y1,0)])/ wbr )
              - ( square(image[imageLinearIndex(x0,y0,1)] - image[imageLinearIndex(x1,y1,1)])/ wbg )
              - ( square(image[imageLinearIndex(x0,y0,2)] - image[imageLinearIndex(x1,y1,2)])/ wbb )
          );
  }

  float gaussian_cost(int x0, int y0, int x1, int y1)
  {
    return (float)exp(
            - ( square(x0-x1)/ wgx ) 
            - ( square(y0-y1)/ wgy ) 
          );
  }
  private:
    const unsigned char * image;
    LinearIndex imageLinearIndex;

    // Scaling constnat for (B)ilteral and (G)aussian weights.
    float wbx;
    float wby;
    float wbr;
    float wbg;
    float wbb;

    float wgx;
    float wgy;

    float bilateral_weight;
    float gaussian_weight;
};

// Calculate energy for DenseCRF solutions
class EnergyFunctor
{
public:
  EnergyFunctor(UnaryCost _unaryCost, 
                PairwiseCost _pairwiseCost,
                int _M, 
                int _N,  
                const int _numberOfLabels) :
                unaryCost(_unaryCost), 
                pairwiseCost(_pairwiseCost),
                M(_M), 
                N(_N), 
                numberOfLabels(_numberOfLabels)
  {
    numVariables = N*M;
  };

  double operator() (short * map)
  {
    Linear2sub linear2sub(M,N);

    double cost = 0;
    
    // Unary cost
    for (int i = 0; i < numVariables; i++)
    {
      std::pair<int,int> p = linear2sub(i);
      cost += unaryCost(p ,map[i]); 
    }

    // Pairwise cost
    for (int i = 0; i < numVariables; i++) {
    for (int j = i+1; j < numVariables; j++) {

      // // Potts defined as meanfield library
      // // -cost when label i = label j 
      // if (map[i] == map[j])
      // {
      //   std::pair<int,int> p0 = linear2sub(i);
      //   std::pair<int,int> p1 = linear2sub(j);

      //   cost -= pairwiseCost(p0,p1);
      // }

      // Usual definition
      if (map[i] != map[j])
      {
        std::pair<int,int> p0 = linear2sub(i);
        std::pair<int,int> p1 = linear2sub(j);

        cost += pairwiseCost(p0,p1);
      }
    }
    }

    return cost;
  }

private:
    UnaryCost unaryCost;
    PairwiseCost pairwiseCost;

    const int M;
    const int N;
    int numVariables;
    const size_t numberOfLabels;
};

// Goes through each pixel and choices the lowest unary cost
// this can be used as a very crude lower bound
double lowestUnaryCost(const float * unary, int M, int N, int numberOfLabels)
{
  LinearIndex unaryLinearIndex(M,N, numberOfLabels);
  double cost = 0;

  for (int x = 0; x < M; x++) {
  for (int y = 0; y < N; y++) {

    double low =  unary[unaryLinearIndex(x,y,0)];

    for (int l = 1; l < numberOfLabels; l++) 
    {
        if (low > unary[unaryLinearIndex(x,y,l)])
          low = unary[unaryLinearIndex(x,y,l)];
    }

    cost += low;
  }
  }

  return cost;
}

void threshold_solution(short * map,  const float * unary, int M, int N, int numberOfLabels)
{
 // Calculate energy for start guess 
  LinearIndex TwoDIndex(M,N,1);
  LinearIndex unaryLinearIndex(M,N, numberOfLabels);

  for (int x = 0; x < M; x++) {
  for (int y = 0; y < N; y++) {

    double low =  unary[unaryLinearIndex(x,y,0)];
    map[TwoDIndex(x,y,0)] = 0;

    for (int l = 1; l < numberOfLabels; l++) 
    {
        if (low > unary[unaryLinearIndex(x,y,l)])
        {
          low = unary[unaryLinearIndex(x,y,l)];
          map[TwoDIndex(x,y,0)] = l;
        }
    }
  }
  }
}

void normalize(matrix<double>& Q)
{
  double sum = 0;
  for (int x =0; x < Q.M; x++) {
  for (int y =0; y < Q.N; y++) {

    sum = Q(x,y,0);
    for (int l = 1; l < Q.O; l++)
      sum += Q(x,y,l);

    for (int l = 0; l < Q.O; l++)
      Q(x,y,l) /= sum;
  } 
  }

  return;
}
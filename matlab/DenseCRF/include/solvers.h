#pragma once

#include "TRW_S-v1.3/MRFEnergy.h"
typedef MRFEnergy<TypePotts> TRWS;
typedef MRFEnergy<TypePotts>::NodeId TRWSNodes;
typedef MRFEnergy<TypePotts>::Options TRWSOptions;

#include "maxflow-v3.03.src/graph.h"
typedef Graph<float,float,float> GraphType;

#include "extendedDenseCRF2D.h"
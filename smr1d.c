

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define FISH_PRIVATE_DEFS
#include "fish.h"


fish_block *fish_block_new()
{
  fish_block *B = (fish_block*) malloc(sizeof(fish_block));
  fish_block block = {
    .rank = 1,
    .guard = 0,
    .size = { 1, 1, 1 },
    .fluid = NULL,
    .descr = NULL,
    .error = NULL,
  } ;
  *B = block;
  return B;
}

int fish_block_del(fish_block *B)
{
  int ntot = fish_block_totalstates(B);
  if (B->fluid) {
    for (int n=0; n<ntot; ++n) {
      fluids_state_del(B->fluid[n]);
    }
    free(B->fluid);
    B->fluid = NULL;
  }
  free(B);
  return 0;
}

char *fish_block_geterror(fish_block *B)
{
  return B->error;
}

int fish_block_getsize(fish_block *B, int dim)
{
  if (dim < B->rank) {
    return B->size[dim];
  }
  else {
    B->error = "argument 'dim' must be smaller than the rank of the block";
    return FISH_ERROR;
  }
}

int fish_block_setsize(fish_block *B, int dim, int size)
{
  if (dim < B->rank) {
    B->size[dim] = size;
    return 0;
  }
  else {
    B->error = "argument 'dim' must be smaller than the rank of the block";
    return FISH_ERROR;
  }
}

int fish_block_getrank(fish_block *B)
{
  return B->rank;
}

int fish_block_setrank(fish_block *B, int rank)
{
  if (rank >= 1 && rank <= 3) {
    B->rank = rank;
    return 0;
  }
  else {
    B->error = "rank must be 1, 2, or 3";
    return FISH_ERROR;
  }
}

int fish_block_getguard(fish_block *B)
{
  return B->guard;
}

int fish_block_setguard(fish_block *B, int guard)
{
  B->guard = guard;
  return 0;
}

int fish_block_getdescr(fish_block *B, fluids_descr **D)
{
  *D = B->descr;
  return 0;
}

int fish_block_setdescr(fish_block *B, fluids_descr *D)
{
  B->descr = D;
  return 0;
}

int fish_block_totalstates(fish_block *B)
{
  int ng = B->guard;
  switch (B->rank) {
  case 1: return (B->size[0]+2*ng);
  case 2: return (B->size[0]+2*ng) * (B->size[1]+2*ng);
  case 3: return (B->size[0]+2*ng) * (B->size[1]+2*ng) * (B->size[2]+2*ng);
  default: return 0;
  }
}

int fish_block_allocate(fish_block *B)
{
  if (B->descr == NULL) {
    B->error = "block's fluid descriptor must be set before allocating";
    return FISH_ERROR;
  }

  int ntot = fish_block_totalstates(B);
  B->fluid = (fluids_state**) realloc(B->fluid, ntot * sizeof(fluids_state*));
  for (int n=0; n<ntot; ++n) {
    B->fluid[n] = fluids_state_new();
    fluids_state_setdescr(B->fluid[n], B->descr);
  }

  return 0;
}

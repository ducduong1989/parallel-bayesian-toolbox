
// Includes all PBT header files

#ifndef PARALLEL_BAYES_TOOLBOX
#define PARALLEL_BAYES_TOOLBOX

#include "ppf_helper_functions.h"
#include "particle.h"
#include "particles.h"

// Bayes filters
#include "pf.h"
#include "pf_sir.h"
#include "pf_rpf.h"
#include "pf_ekf.h"
#include "pf_ukf.h"
#include "pf_kld.h"

// Model definition
#include "model.h"

// Noise Batch
#include "noise_batch.h"

// Noise source
#include "noise.h"

// State Estimation
#include "estimation.h"
#include "estimation_mean.h"
#include "estimation_kmeans.h"
#include "estimation_mean_shift.h"
#include "estimation_median.h"
#include "estimation_mode.h"
#include "estimation_modema.h"
#include "estimation_best_particle.h"
#include "estimation_robust_mean.h"

// Resampling
#include "resampling.h"
#include "resampling_multinomial.h"
#include "resampling_residual.h"
#include "resampling_systematic.h"
#include "resampling_stratified.h"
#include "resampling_naive_delete.h"

// Noise Sources
#include "noises/ppfnoises.h"

#endif

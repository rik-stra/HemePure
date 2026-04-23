// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_QOI_QOITRACKINGACTOR_H
#define HEMELB_LB_QOI_QOITRACKINGACTOR_H

#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "configuration/SimConfig.h"
#include "geometry/LatticeData.h"
#include "lb/MacroscopicPropertyCache.h"
#include "lb/SimulationState.h"
#include "lb/qoi/KernelScaleAwareQoi.h"
#include "net/IteratedAction.h"
#include "net/IOCommunicator.h"
#include "util/UnitConverter.h"
#include "reporting/Timers.h"

namespace hemelb
{
  namespace lb
  {
    namespace qoi
    {
      // Non-templated tracking actor.  The caller provides an EDM correction callback that
      // maps to LBM<LatticeType>::ApplyEDMCorrections so this class stays lattice-agnostic.
      class QoiTrackingActor : public net::IteratedAction
      {
        public:
          using EDMCallback = std::function<void(const std::vector<util::Vector3D<distribn_t> >&)>;

          QoiTrackingActor(const lb::SimulationState& simulationState,
                           const geometry::LatticeData& latticeData,
                           lb::MacroscopicPropertyCache& propertyCache,
                           const net::IOCommunicator& ioComms,
                           const util::UnitConverter& unitConverter,
                           reporting::Timers& timers,
                           const configuration::SimConfig::QoiTrackingConfig& config,
                           const std::string& tauOutputPath,
                           EDMCallback edmCallback);
          ~QoiTrackingActor();

          void SetRequiredProperties();
          void EndIteration();

        private:
          bool ShouldTrack(unsigned long timestep) const;

          // Load reference CSV produced by a high-fidelity run.
          // Columns: timestep, E_G3, E_L3, E_L1, Z_G3, Z_L3, Z_L1 (physical units).
          void LoadReferenceTrajectory();

          // MPI gather of downsampled velocities (same as KernelQoiActor).
          void CacheCoarseLocalSiteIndices();
          void InitializeDownsampledExchangeLayout();
          std::unordered_map<site_t, util::Vector3D<distribn_t> > BuildGlobalDownsampledVelocityCache() const;

          // Solve a dense N×N linear system A x = b (in-place; modifies A).
          // Returns false if singular.
          static bool SolveLinearSystem(std::vector<double>& A, std::vector<double>& b,
                                        int N, std::vector<double>& x);

          // Compute tau_i and the per-coarse-site correction field, apply EDM correction.
          void ComputeAndApplyCorrection(
              const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloCache,
              unsigned long timestep);

          const lb::SimulationState& simulationState;
          const geometry::LatticeData& latticeData;
          lb::MacroscopicPropertyCache& propertyCache;
          const net::IOCommunicator& ioComms;
          const util::UnitConverter& unitConverter;
          reporting::Timers& timers;
          configuration::SimConfig::QoiTrackingConfig config;
          std::string tauOutputPath;
          EDMCallback edmCallback;

          KernelScaleAwareQoiCalculator calculator;
          std::vector<KernelDefinition> kernels;

          // Reference QoI trajectories in PHYSICAL units.
          // refTrajectory[timestep] -> vector of length 2*nKernels: [E_k..., Z_k...]
          std::unordered_map<unsigned long, std::vector<double> > refTrajectory;

          // Physical scaling factors (set once in constructor).
          double energyScale;
          double enstrophyScale;

          // MPI exchange layout (same as KernelQoiActor).
          std::vector<site_t> coarseLocalSiteIndices;
          std::vector<site_t> coarseLocalGlobalIds;
          std::vector<int> gatherCounts;
          std::vector<int> gatherDisplacements;
          std::vector<int> velocityCounts;
          std::vector<int> velocityDisplacements;
          int totalGatheredCount;
          std::vector<site_t> gatheredIds;
          mutable std::vector<distribn_t> localVelocities;
          mutable std::vector<distribn_t> gatheredVelocities;

          std::ofstream tauStream;
          unsigned long writesSinceLastFlush;
      };
    }
  }
}

#endif // HEMELB_LB_QOI_QOITRACKINGACTOR_H

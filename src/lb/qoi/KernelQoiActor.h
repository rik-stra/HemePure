// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_QOI_KERNELQOIACTOR_H
#define HEMELB_LB_QOI_KERNELQOIACTOR_H

#include <fstream>
#include <string>
#include <vector>

#include "configuration/SimConfig.h"
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
      class KernelQoiActor : public net::IteratedAction
      {
        public:
          KernelQoiActor(const lb::SimulationState& simulationState,
                         const geometry::LatticeData& latticeData,
                         lb::MacroscopicPropertyCache& propertyCache,
                         const net::IOCommunicator& ioComms,
                         const util::UnitConverter& unitConverter,
                         reporting::Timers& timers,
                         const configuration::SimConfig::KernelQoiOutputConfig& config,
                         const std::string& outputPath);

          void SetRequiredProperties();
          void EndIteration();

        private:
          bool ShouldWrite(unsigned long timestepNumber) const;
          void InitializeOutputFile();
          void CacheCoarseLocalSiteIndices();
          void InitializeDownsampledExchangeLayout();
          std::unordered_map<site_t, util::Vector3D<distribn_t> > BuildGlobalDownsampledVelocityCache() const;

          const lb::SimulationState& simulationState;
          const geometry::LatticeData& latticeData;
          lb::MacroscopicPropertyCache& propertyCache;
          const net::IOCommunicator& ioComms;
          const util::UnitConverter& unitConverter;
          reporting::Timers& timers;
          configuration::SimConfig::KernelQoiOutputConfig config;
          std::string outputPath;
          std::ofstream outputStream;

          KernelScaleAwareQoiCalculator calculator;
          std::vector<KernelDefinition> kernels;
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
      };
    }
  }
}

#endif // HEMELB_LB_QOI_KERNELQOIACTOR_H

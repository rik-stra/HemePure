// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/qoi/KernelQoiActor.h"

#include <fstream>
#include <unordered_map>
#include <vector>

#include "log/Logger.h"
#include "net/MpiDataType.h"
#include "net/mpi.h"

namespace hemelb
{
  namespace lb
  {
    namespace qoi
    {
      KernelQoiActor::KernelQoiActor(const lb::SimulationState& simulationState,
                                     const geometry::LatticeData& latticeData,
                                     lb::MacroscopicPropertyCache& propertyCache,
                                     const net::IOCommunicator& ioComms,
                                     const util::UnitConverter& unitConverter,
                                     reporting::Timers& timers,
                                     const configuration::SimConfig::KernelQoiOutputConfig& config,
                                     const std::string& outputPath) :
        simulationState(simulationState),
        latticeData(latticeData),
        propertyCache(propertyCache),
        ioComms(ioComms),
        unitConverter(unitConverter),
        timers(timers),
        config(config),
        outputPath(outputPath),
        calculator(latticeData, config.coarseningFactor),
        kernels(calculator.CreateDefaultKernels()),
        totalGatheredCount(0)
      {
        InitializeOutputFile();
        CacheCoarseLocalSiteIndices();
        InitializeDownsampledExchangeLayout();

        if (ioComms.Size() > 1)
        {
          log::Logger::Log<log::Warning, log::Singleton>(
              "Kernel QoI output exchanges only the downsampled velocity field across ranks before stencil evaluation."
              " Only out-of-domain samples are zero-padded.");
        }
      }

      bool KernelQoiActor::ShouldWrite(unsigned long timestepNumber) const
      {
        if (!config.enabled)
        {
          return false;
        }

        return (timestepNumber >= config.start) && (timestepNumber <= config.stop)
            && ((timestepNumber % config.frequency) == 0);
      }

      void KernelQoiActor::SetRequiredProperties()
      {
        if (ShouldWrite(simulationState.GetTimeStep()))
        {
          propertyCache.velocityCache.SetRefreshFlag();
        }
      }

      void KernelQoiActor::InitializeOutputFile()
      {
        if (!config.enabled || !ioComms.OnIORank())
        {
          return;
        }

        outputStream.open(outputPath.c_str(), std::ios::out | std::ios::trunc);
        outputStream << "timestep";
        for (std::size_t i = 0; i < kernels.size(); ++i)
        {
          outputStream << ",E_" << kernels[i].label;
        }
        for (std::size_t i = 0; i < kernels.size(); ++i)
        {
          outputStream << ",Z_" << kernels[i].label;
        }
        outputStream << "\n";
      }

      void KernelQoiActor::CacheCoarseLocalSiteIndices()
      {
        const site_t localSiteCount = latticeData.GetLocalFluidSiteCount();
        const int coarseningFactor = config.coarseningFactor;

        for (site_t i = 0; i < localSiteCount; ++i)
        {
          util::Vector3D<site_t> globalCoords = latticeData.GetSiteCoordsFromSiteId(i);
          if (coarseningFactor == 1 || ((globalCoords.x % coarseningFactor) == 0 &&
                                        (globalCoords.y % coarseningFactor) == 0 &&
                                        (globalCoords.z % coarseningFactor) == 0))
          {
            coarseLocalSiteIndices.push_back(i);
            coarseLocalGlobalIds.push_back(latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(globalCoords));
          }
        }

        localVelocities.resize(3 * coarseLocalSiteIndices.size());
      }

      void KernelQoiActor::InitializeDownsampledExchangeLayout()
      {
        if (!config.enabled)
        {
          return;
        }

        gatherCounts.assign(ioComms.Size(), 0);
        gatherDisplacements.assign(ioComms.Size(), 0);
        velocityCounts.assign(ioComms.Size(), 0);
        velocityDisplacements.assign(ioComms.Size(), 0);

        const int localCount = static_cast<int>(coarseLocalGlobalIds.size());
        HEMELB_MPI_CALL(MPI_Allgather,
                        (&localCount, 1, MPI_INT,
                         &gatherCounts[0], 1, MPI_INT,
                         ioComms));

        totalGatheredCount = 0;
        for (int i = 0; i < ioComms.Size(); ++i)
        {
          gatherDisplacements[i] = totalGatheredCount;
          totalGatheredCount += gatherCounts[i];
          velocityCounts[i] = 3 * gatherCounts[i];
          velocityDisplacements[i] = 3 * gatherDisplacements[i];
        }

        if (totalGatheredCount == 0)
        {
          return;
        }

        gatheredIds.resize(totalGatheredCount);
        gatheredVelocities.resize(3 * totalGatheredCount);

        HEMELB_MPI_CALL(MPI_Allgatherv,
                        (localCount > 0 ? &coarseLocalGlobalIds[0] : NULL,
                         localCount, net::MpiDataType<site_t>(),
                         &gatheredIds[0], &gatherCounts[0], &gatherDisplacements[0], net::MpiDataType<site_t>(),
                         ioComms));
      }

      std::unordered_map<site_t, util::Vector3D<distribn_t> > KernelQoiActor::BuildGlobalDownsampledVelocityCache() const
      {
        std::unordered_map<site_t, util::Vector3D<distribn_t> > haloVelocityCache;

        if (totalGatheredCount == 0)
        {
          return haloVelocityCache;
        }

        for (std::size_t j = 0; j < coarseLocalSiteIndices.size(); ++j)
        {
          const site_t i = coarseLocalSiteIndices[j];
          const util::Vector3D<distribn_t>& velocity = propertyCache.velocityCache.Get(i);
          localVelocities[3 * j + 0] = velocity.x;
          localVelocities[3 * j + 1] = velocity.y;
          localVelocities[3 * j + 2] = velocity.z;
        }

        HEMELB_MPI_CALL(MPI_Allgatherv,
            (coarseLocalSiteIndices.size() > 0 ? &localVelocities[0] : NULL,
                         static_cast<int>(3 * coarseLocalSiteIndices.size()), net::MpiDataType<distribn_t>(),
                         &gatheredVelocities[0], &velocityCounts[0], &velocityDisplacements[0], net::MpiDataType<distribn_t>(),
                         ioComms));

        haloVelocityCache.reserve(static_cast<std::size_t>(totalGatheredCount));
        for (int index = 0; index < totalGatheredCount; ++index)
        {
          haloVelocityCache[gatheredIds[index]] = util::Vector3D<distribn_t>(
              gatheredVelocities[3 * index + 0],
              gatheredVelocities[3 * index + 1],
              gatheredVelocities[3 * index + 2]);
        }

        return haloVelocityCache;
      }

      void KernelQoiActor::EndIteration()
      {
        if (!ShouldWrite(simulationState.GetTimeStep()))
        {
          return;
        }

        timers[reporting::Timers::extractionWriting].Start();

        std::unordered_map<site_t, util::Vector3D<distribn_t> > haloVelocityCache = BuildGlobalDownsampledVelocityCache();

        std::vector<double> localEnergy;
        std::vector<double> localEnstrophy;
        calculator.ComputeLocalIntegrals(propertyCache, haloVelocityCache, kernels, localEnergy, localEnstrophy);

        std::vector<double> localPacked;
        localPacked.reserve(localEnergy.size() + localEnstrophy.size());
        for (std::size_t i = 0; i < localEnergy.size(); ++i)
        {
          localPacked.push_back(localEnergy[i]);
        }
        for (std::size_t i = 0; i < localEnstrophy.size(); ++i)
        {
          localPacked.push_back(localEnstrophy[i]);
        }

        std::vector<double> globalPacked = ioComms.AllReduce(localPacked, MPI_SUM);

        const PhysicalDistance voxelSize = unitConverter.GetVoxelSize();
        const PhysicalDistance coarseVoxelSize = voxelSize * static_cast<double>(config.coarseningFactor);
        const PhysicalSpeed velocityScale = unitConverter.ConvertVelocityToPhysicalUnits(1.0);
        const double energyScale = velocityScale * velocityScale * coarseVoxelSize * coarseVoxelSize * coarseVoxelSize;
        const double enstrophyScale = velocityScale * velocityScale * coarseVoxelSize
            * static_cast<double>(config.coarseningFactor)
            * static_cast<double>(config.coarseningFactor);

        for (std::size_t i = 0; i < kernels.size(); ++i)
        {
          globalPacked[i] *= energyScale;
          globalPacked[i + kernels.size()] *= enstrophyScale;
        }

        if (ioComms.OnIORank())
        {
          outputStream << simulationState.GetTimeStep();
          for (std::size_t i = 0; i < globalPacked.size(); ++i)
          {
            outputStream << "," << globalPacked[i];
          }
          outputStream << "\n";
          outputStream.flush();
        }

        timers[reporting::Timers::extractionWriting].Stop();
      }
    }
  }
}

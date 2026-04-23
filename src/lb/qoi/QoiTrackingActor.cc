// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/qoi/QoiTrackingActor.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "log/Logger.h"
#include "net/MpiDataType.h"
#include "net/mpi.h"

namespace hemelb
{
  namespace lb
  {
    namespace qoi
    {
      QoiTrackingActor::QoiTrackingActor(
          const lb::SimulationState& simulationState,
          const geometry::LatticeData& latticeData,
          lb::MacroscopicPropertyCache& propertyCache,
          const net::IOCommunicator& ioComms,
          const util::UnitConverter& unitConverter,
          reporting::Timers& timers,
          const configuration::SimConfig::QoiTrackingConfig& config,
          const std::string& tauOutputPath,
          EDMCallback edmCallback) :
        simulationState(simulationState),
        latticeData(latticeData),
        propertyCache(propertyCache),
        ioComms(ioComms),
        unitConverter(unitConverter),
        timers(timers),
        config(config),
        tauOutputPath(tauOutputPath),
        edmCallback(edmCallback),
        calculator(latticeData, config.coarseningFactor),
        kernels(calculator.CreateDefaultKernels()),
        energyScale(1.0),
        enstrophyScale(1.0),
        totalGatheredCount(0),
        writesSinceLastFlush(0)
      {
        // Compute physical scaling factors (same formulae as KernelQoiActor).
        const PhysicalDistance voxelSize = unitConverter.GetVoxelSize();
        const PhysicalDistance coarseVoxelSize = voxelSize * static_cast<double>(config.coarseningFactor);
        const PhysicalSpeed velocityScale = unitConverter.ConvertVelocityToPhysicalUnits(1.0);
        energyScale = velocityScale * velocityScale * coarseVoxelSize * coarseVoxelSize * coarseVoxelSize;
        enstrophyScale = velocityScale * velocityScale * coarseVoxelSize
            * static_cast<double>(config.coarseningFactor)
            * static_cast<double>(config.coarseningFactor);

        LoadReferenceTrajectory();
        CacheCoarseLocalSiteIndices();
        InitializeDownsampledExchangeLayout();

        if (ioComms.OnIORank() && !tauOutputPath.empty())
        {
          tauStream.open(tauOutputPath.c_str(), std::ios::out | std::ios::trunc);
          tauStream << "timestep";
          for (std::size_t k = 0; k < kernels.size(); ++k)
          {
            tauStream << ",dQ_E_" << kernels[k].label;
          }
          for (std::size_t k = 0; k < kernels.size(); ++k)
          {
            tauStream << ",dQ_Z_" << kernels[k].label;
          }
          tauStream << "\n";
        }
      }

      QoiTrackingActor::~QoiTrackingActor()
      {
        if (tauStream.is_open())
        {
          tauStream.flush();
        }
      }

      bool QoiTrackingActor::ShouldTrack(unsigned long timestep) const
      {
        if (!config.enabled)
        {
          return false;
        }
        return (timestep >= config.start) && (timestep <= config.stop)
            && ((timestep % config.frequency) == 0);
      }

      void QoiTrackingActor::LoadReferenceTrajectory()
      {
        if (!config.enabled || config.referenceFile.empty())
        {
          return;
        }

        // Only the IO rank loads from disk; then all ranks get the data via broadcast.
        std::vector<unsigned long> timesteps;
        std::vector<double> values; // interleaved: per-timestep QoI values (length 2*nKernels each)
        const std::size_t nQoi = 2 * kernels.size();

        if (ioComms.OnIORank())
        {
          std::ifstream in(config.referenceFile.c_str());
          if (!in.is_open())
          {
            log::Logger::Log<log::Warning, log::Singleton>(
                "QoiTrackingActor: cannot open reference file '%s'",
                config.referenceFile.c_str());
            return;
          }

          std::string line;
          std::getline(in, line); // skip header
          while (std::getline(in, line))
          {
            if (line.empty() || line[0] == '#')
            {
              continue;
            }
            std::istringstream ss(line);
            unsigned long t;
            char sep;
            ss >> t;
            timesteps.push_back(t);
            for (std::size_t q = 0; q < nQoi; ++q)
            {
              double v = 0.0;
              ss >> sep >> v;
              values.push_back(v);
            }
          }
          log::Logger::Log<log::Info, log::Singleton>(
              "QoiTrackingActor: loaded %zu reference timesteps from '%s'",
              timesteps.size(), config.referenceFile.c_str());
        }

        // Broadcast to all ranks.
        unsigned long nRows = static_cast<unsigned long>(timesteps.size());
        HEMELB_MPI_CALL(MPI_Bcast, (&nRows, 1, net::MpiDataType<unsigned long>(), 0, ioComms));
        if (nRows == 0)
        {
          return;
        }
        timesteps.resize(nRows);
        values.resize(nRows * nQoi);
        HEMELB_MPI_CALL(MPI_Bcast, (timesteps.data(), static_cast<int>(nRows),
                                    net::MpiDataType<unsigned long>(), 0, ioComms));
        HEMELB_MPI_CALL(MPI_Bcast, (values.data(), static_cast<int>(nRows * nQoi),
                                    net::MpiDataType<double>(), 0, ioComms));

        refTrajectory.reserve(nRows);
        for (unsigned long r = 0; r < nRows; ++r)
        {
          std::vector<double> qv(values.begin() + r * nQoi, values.begin() + (r + 1) * nQoi);
          refTrajectory[timesteps[r]] = std::move(qv);
        }
      }

      void QoiTrackingActor::CacheCoarseLocalSiteIndices()
      {
        const site_t localSiteCount = latticeData.GetLocalFluidSiteCount();
        const int cf = config.coarseningFactor;
        for (site_t i = 0; i < localSiteCount; ++i)
        {
          util::Vector3D<site_t> gc = latticeData.GetSiteCoordsFromSiteId(i);
          if (cf == 1 || (gc.x % cf == 0 && gc.y % cf == 0 && gc.z % cf == 0))
          {
            coarseLocalSiteIndices.push_back(i);
            coarseLocalGlobalIds.push_back(
                latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(gc));
          }
        }
        localVelocities.resize(3 * coarseLocalSiteIndices.size());
      }

      void QoiTrackingActor::InitializeDownsampledExchangeLayout()
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
                        (&localCount, 1, MPI_INT, &gatherCounts[0], 1, MPI_INT, ioComms));

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
                         &gatheredIds[0], &gatherCounts[0], &gatherDisplacements[0],
                         net::MpiDataType<site_t>(), ioComms));
      }

      std::unordered_map<site_t, util::Vector3D<distribn_t> >
      QoiTrackingActor::BuildGlobalDownsampledVelocityCache() const
      {
        std::unordered_map<site_t, util::Vector3D<distribn_t> > cache;
        if (totalGatheredCount == 0)
        {
          return cache;
        }
        for (std::size_t j = 0; j < coarseLocalSiteIndices.size(); ++j)
        {
          const site_t i = coarseLocalSiteIndices[j];
          const util::Vector3D<distribn_t>& vel = propertyCache.postCollisionVelocityCache.Get(i);
          localVelocities[3 * j + 0] = vel.x;
          localVelocities[3 * j + 1] = vel.y;
          localVelocities[3 * j + 2] = vel.z;
        }
        HEMELB_MPI_CALL(
            MPI_Allgatherv,
            (coarseLocalSiteIndices.size() > 0 ? &localVelocities[0] : NULL,
             static_cast<int>(3 * coarseLocalSiteIndices.size()), net::MpiDataType<distribn_t>(),
             &gatheredVelocities[0], &velocityCounts[0], &velocityDisplacements[0],
             net::MpiDataType<distribn_t>(), ioComms));

        cache.reserve(static_cast<std::size_t>(totalGatheredCount));
        for (int idx = 0; idx < totalGatheredCount; ++idx)
        {
          cache[gatheredIds[idx]] = util::Vector3D<distribn_t>(
              gatheredVelocities[3 * idx + 0],
              gatheredVelocities[3 * idx + 1],
              gatheredVelocities[3 * idx + 2]);
        }
        return cache;
      }

      // ---- Gauss-Jordan elimination with partial pivoting ----
      bool QoiTrackingActor::SolveLinearSystem(std::vector<double>& A, std::vector<double>& b,
                                               int N, std::vector<double>& x)
      {
        x.assign(N, 0.0);
        for (int col = 0; col < N; ++col)
        {
          // Partial pivot
          int pivot = col;
          for (int row = col + 1; row < N; ++row)
          {
            if (std::abs(A[row * N + col]) > std::abs(A[pivot * N + col]))
            {
              pivot = row;
            }
          }
          if (pivot != col)
          {
            for (int k = 0; k < N; ++k)
            {
              std::swap(A[col * N + k], A[pivot * N + k]);
            }
            std::swap(b[col], b[pivot]);
          }
          if (std::abs(A[col * N + col]) < 1e-30)
          {
            return false;
          }
          const double scale = A[col * N + col];
          for (int k = col; k < N; ++k)
          {
            A[col * N + k] /= scale;
          }
          b[col] /= scale;
          for (int row = 0; row < N; ++row)
          {
            if (row == col)
            {
              continue;
            }
            const double factor = A[row * N + col];
            for (int k = col; k < N; ++k)
            {
              A[row * N + k] -= factor * A[col * N + k];
            }
            b[row] -= factor * b[col];
          }
        }
        x = b;
        return true;
      }

      void QoiTrackingActor::SetRequiredProperties()
      {
        if (ShouldTrack(simulationState.GetTimeStep()))
        {
          propertyCache.postCollisionVelocityCache.SetRefreshFlag();
        }
      }

      void QoiTrackingActor::EndIteration()
      {
        const unsigned long t = simulationState.GetTimeStep();
        if (!ShouldTrack(t))
        {
          return;
        }

        timers[reporting::Timers::extractionWriting].Start();

        auto haloCache = BuildGlobalDownsampledVelocityCache();
        ComputeAndApplyCorrection(haloCache, t);

        timers[reporting::Timers::extractionWriting].Stop();
      }

      void QoiTrackingActor::ComputeAndApplyCorrection(
          const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloCache,
          unsigned long timestep)
      {
        const std::size_t nK = kernels.size();
        const std::size_t nQoi = 2 * nK;

        // ---- Step 1: compute predicted QoIs (lattice units) ----
        std::vector<double> localEnergy, localEnstrophy;
        calculator.ComputeLocalIntegrals(propertyCache, haloCache, kernels, localEnergy, localEnstrophy);

        std::vector<double> localPacked;
        localPacked.insert(localPacked.end(), localEnergy.begin(), localEnergy.end());
        localPacked.insert(localPacked.end(), localEnstrophy.begin(), localEnstrophy.end());
        // globalQoi stays in lattice units — the solve is done in lattice units.
        std::vector<double> globalQoi = ioComms.AllReduce(localPacked, MPI_SUM);

        // ---- Step 2: look up reference QoIs (physical units) ----
        // Physical-time alignment: HF step = LF step * timestepStride, so when dt_LF = s * dt_HF
        // the reference CSV row for physical time t^n is read at raw HF key = n * s.
        const unsigned long refKey = timestep * config.timestepStride;
        auto refIt = refTrajectory.find(refKey);
        if (refIt == refTrajectory.end())
        {
          return;
        }
        const std::vector<double>& refQoi = refIt->second;
        if (refQoi.size() != nQoi)
        {
          return;
        }

        // dQ in lattice units: the solve is done entirely in lattice units so that the
        // Gram matrix (also in lattice units) and dQ are consistent.
        // globalQoi is currently in lattice units (before physical scaling).
        std::vector<double> dQLattice(nQoi);
        for (std::size_t k = 0; k < nK; ++k)
        {
          dQLattice[k]      = refQoi[k]      / energyScale    - globalQoi[k];
          dQLattice[k + nK] = refQoi[k + nK] / enstrophyScale - globalQoi[k + nK];
        }

        // ---- Step 3: compute functional derivative fields and Gram matrix ----
        std::vector<std::vector<util::Vector3D<distribn_t> > > energyFDs, enstrophyFDs;
        std::vector<double> localGram;
        calculator.ComputeLocalFDFieldsAndGram(propertyCache, haloCache, kernels,
                                               energyFDs, enstrophyFDs, localGram);

        std::vector<double> gram = ioComms.AllReduce(localGram, MPI_SUM);

        // ---- Step 4: orthogonalize patterns and compute influence coefficients ----
        // Patterns O_i = V_i + sum_{j!=i} c_{ij} V_j.
        // Orthogonality: <V_l, O_i> = 0 for l != i.
        // For each i, solve (N-1)x(N-1) system for off-diagonal c_{ij}.
        // Influence: sigma_i = <V_i, O_i> = gram[i,i] + sum_{j!=i} c_{ij} gram[i,j].
        std::vector<double> sigma(nQoi, 0.0);
        // c coefficients stored as c[i * nQoi + j] (c[i,i] = 1, others from solve).
        std::vector<double> c(nQoi * nQoi, 0.0);
        for (std::size_t qi = 0; qi < nQoi; ++qi)
        {
          c[qi * nQoi + qi] = 1.0;
        }

        for (std::size_t qi = 0; qi < nQoi; ++qi)
        {
          // Build (nQoi-1) x (nQoi-1) system for c[qi, j!=qi].
          const int Nsub = static_cast<int>(nQoi) - 1;
          std::vector<double> Asub(Nsub * Nsub, 0.0);
          std::vector<double> bsub(Nsub, 0.0);

          // Map sub-indices (skip qi).
          std::vector<std::size_t> idx;
          for (std::size_t q = 0; q < nQoi; ++q)
          {
            if (q != qi)
            {
              idx.push_back(q);
            }
          }
          for (int li = 0; li < Nsub; ++li)
          {
            const std::size_t l = idx[li];
            bsub[li] = -gram[l * nQoi + qi];
            for (int ki = 0; ki < Nsub; ++ki)
            {
              const std::size_t jj = idx[ki];
              Asub[li * Nsub + ki] = gram[l * nQoi + jj];
            }
          }

          std::vector<double> csub;
          if (Nsub > 0 && SolveLinearSystem(Asub, bsub, Nsub, csub))
          {
            for (int ki = 0; ki < Nsub; ++ki)
            {
              c[qi * nQoi + idx[ki]] = csub[ki];
            }
          }

          // Influence: sigma_i = sum_j c[i,j] * gram[i,j].
          double sig = 0.0;
          for (std::size_t qj = 0; qj < nQoi; ++qj)
          {
            sig += c[qi * nQoi + qj] * gram[qi * nQoi + qj];
          }
          sigma[qi] = sig;
        }

        // ---- Step 5: compute tau_i = dQ_i / sigma_i ----
        std::vector<double> tau(nQoi, 0.0);
        for (std::size_t qi = 0; qi < nQoi; ++qi)
        {
          if (std::abs(sigma[qi]) > 1e-30)
          {
            tau[qi] = dQLattice[qi] / sigma[qi];
          }
        }

        // ---- Step 6: compute per-coarse-site correction DeltaV_SGS = sum_i tau_i O_i ----
        const std::size_t nCoarse = calculator.GetCoarseSiteCount();
        std::vector<util::Vector3D<distribn_t> > perCoarseSiteDeltaV(nCoarse,
            util::Vector3D<distribn_t>(0.0, 0.0, 0.0));

        for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
        {
          util::Vector3D<distribn_t> deltaV(0.0, 0.0, 0.0);
          // DeltaV_SGS = sum_{qi=0}^{nQoi-1} tau[qi] * O_qi(x)
          // O_qi(x) = sum_{qj} c[qi,qj] * V_qj(x)
          // V_qj for qj in [0,nK): energy FD for kernel qj
          // V_qj for qj in [nK,2nK): enstrophy FD for kernel qj-nK
          for (std::size_t qi = 0; qi < nQoi; ++qi)
          {
            if (tau[qi] == 0.0)
            {
              continue;
            }
            for (std::size_t qj = 0; qj < nK; ++qj)
            {
              const double w = tau[qi] * c[qi * nQoi + qj];
              const util::Vector3D<distribn_t>& vj = energyFDs[qj][siteIndex];
              deltaV.x += w * vj.x;
              deltaV.y += w * vj.y;
              deltaV.z += w * vj.z;
            }
            for (std::size_t qj = 0; qj < nK; ++qj)
            {
              const double w = tau[qi] * c[qi * nQoi + (nK + qj)];
              const util::Vector3D<distribn_t>& vj = enstrophyFDs[qj][siteIndex];
              deltaV.x += w * vj.x;
              deltaV.y += w * vj.y;
              deltaV.z += w * vj.z;
            }
          }
          perCoarseSiteDeltaV[siteIndex] = deltaV;
        }

        // ---- Step 7: apply EDM correction via callback ----
        std::vector<util::Vector3D<distribn_t> > perSiteDeltaV =
            calculator.BuildPerSiteDeltaV(perCoarseSiteDeltaV);
        edmCallback(perSiteDeltaV);

        // ---- Step 8: write dQ (physical units) to output ----
        // Convert dQLattice back to physical units for interpretable output.
        std::vector<double> dQPhys(nQoi);
        for (std::size_t k = 0; k < nK; ++k)
        {
          dQPhys[k]      = dQLattice[k]      * energyScale;
          dQPhys[k + nK] = dQLattice[k + nK] * enstrophyScale;
        }

        if (ioComms.OnIORank() && tauStream.is_open())
        {
          tauStream << timestep;
          for (std::size_t q = 0; q < nQoi; ++q)
          {
            tauStream << "," << dQPhys[q];
          }
          tauStream << "\n";

          ++writesSinceLastFlush;
          if (config.flushInterval > 0 && writesSinceLastFlush >= config.flushInterval)
          {
            tauStream.flush();
            writesSinceLastFlush = 0;
          }
        }
      }
    }
  }
}

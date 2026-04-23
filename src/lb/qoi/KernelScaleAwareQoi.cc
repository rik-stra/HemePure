// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/qoi/KernelScaleAwareQoi.h"

#include <cmath>

namespace hemelb
{
  namespace lb
  {
    namespace qoi
    {
      namespace
      {
        inline util::Vector3D<distribn_t> ZeroVector()
        {
          return util::Vector3D<distribn_t>(0.0, 0.0, 0.0);
        }

        inline util::Vector3D<distribn_t> AddScaled(const util::Vector3D<distribn_t>& a,
                                                    const util::Vector3D<distribn_t>& b,
                                                    double w)
        {
          return util::Vector3D<distribn_t>(a.x + w * b.x, a.y + w * b.y, a.z + w * b.z);
        }
      }

      KernelScaleAwareQoiCalculator::KernelScaleAwareQoiCalculator(const geometry::LatticeData& latticeData,
                                                                   int coarseningFactor) :
        localGlobalCoords(latticeData.GetLocalFluidSiteCount()),
        globalToLocal(),
        coarseSiteIndices(),
        enstrophyNeighborRefs(),
        filteredWorkspace(),
        tempXWorkspace(),
        tempYWorkspace(),
        remoteTempXCache(),
        remoteTempYCache(),
        remoteFilteredCache(),
        latticeData(latticeData),
        coarseningFactor(coarseningFactor)
      {
        const site_t siteCount = latticeData.GetLocalFluidSiteCount();
        globalToLocal.reserve(siteCount);
        for (site_t i = 0; i < siteCount; ++i)
        {
          const util::Vector3D<site_t>& coord = latticeData.GetSite(i).GetGlobalSiteCoords();
          localGlobalCoords[i] = coord;
          const site_t globalId = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(coord);
          globalToLocal[globalId] = i;
        }

        BuildCoarseSiteIndices();
        BuildEnstrophyNeighborRefs();
      }

      std::vector<KernelDefinition> KernelScaleAwareQoiCalculator::CreateDefaultKernels() const
      {
        std::vector<KernelDefinition> kernels;
        kernels.push_back(CreateGaussianKernel("G3", 3));
        kernels.push_back(CreateLaplacianKernel("L3", 3));
        kernels.push_back(CreateLaplacianKernel("L1", 1));
        return kernels;
      }

      KernelDefinition KernelScaleAwareQoiCalculator::CreateGaussianKernel(const std::string& label, int scale)
      {
        KernelDefinition kernel;
        kernel.label = label;
        kernel.type = GaussianKernel;
        kernel.scale = scale;
        kernel.radius = 4 * scale;

        const double s = static_cast<double>(scale);
        double normalizer = 0.0;

        for (int dx = -kernel.radius; dx <= kernel.radius; ++dx)
        {
          for (int dy = -kernel.radius; dy <= kernel.radius; ++dy)
          {
            for (int dz = -kernel.radius; dz <= kernel.radius; ++dz)
            {
              KernelTap tap;
              tap.dx = dx;
              tap.dy = dy;
              tap.dz = dz;
              tap.weight = std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s * s));
              normalizer += tap.weight;
              kernel.taps.push_back(tap);
            }
          }
        }

        for (std::size_t i = 0; i < kernel.taps.size(); ++i)
        {
          kernel.taps[i].weight /= normalizer;
        }

        return kernel;
      }

      KernelDefinition KernelScaleAwareQoiCalculator::CreateLaplacianKernel(const std::string& label, int scale)
      {
        // 3D analogue of the Laplacian kernel defined in Eq. (L_s) of main.tex:
        //   L_s = (1 / (24 s^3)) * kron_3D(B_3D, J_s^3),
        // where B_3D is the 5x5x5 star-shaped base stencil with nonzero entries
        //   (0,0,0)       -> +6
        //   (+/-1,0,0), (0,+/-1,0), (0,0,+/-1) -> -2  (six axial neighbours at 1)
        //   (+/-2,0,0), (0,+/-2,0), (0,0,+/-2) -> +1  (six axial neighbours at 2)
        // and J_s is the s x s x s block of ones. Sum(B_3D) = 0 (zero-mean),
        // L1 norm(B_3D) = 24, so L_s is unit-L1 after normalization by 24 s^3.
        // For s = 1 this reduces to the 13-tap base stencil.
        // Only odd scales are supported so the Kronecker block has a centered
        // interpretation around each base position.
        KernelDefinition kernel;
        kernel.label = label;
        kernel.type = LaplacianKernel;
        kernel.scale = scale;
        const int blockHalf = (scale - 1) / 2;
        kernel.radius = 2 * scale + blockHalf;

        struct BaseEntry
        {
          int i;
          int j;
          int k;
          double w;
        };
        const BaseEntry base[] = {
          {0, 0, 0, 6.0},
          {1, 0, 0, -2.0},  {-1, 0, 0, -2.0},
          {0, 1, 0, -2.0},  {0, -1, 0, -2.0},
          {0, 0, 1, -2.0},  {0, 0, -1, -2.0},
          {2, 0, 0, 1.0},   {-2, 0, 0, 1.0},
          {0, 2, 0, 1.0},   {0, -2, 0, 1.0},
          {0, 0, 2, 1.0},   {0, 0, -2, 1.0},
        };

        const double s_cubed = static_cast<double>(scale)
                             * static_cast<double>(scale)
                             * static_cast<double>(scale);
        const double invNorm = 1.0 / (24.0 * s_cubed);

        for (const BaseEntry& b : base)
        {
          for (int bi = 0; bi < scale; ++bi)
          {
            for (int bj = 0; bj < scale; ++bj)
            {
              for (int bk = 0; bk < scale; ++bk)
              {
                KernelTap tap;
                tap.dx = scale * b.i + bi - blockHalf;
                tap.dy = scale * b.j + bj - blockHalf;
                tap.dz = scale * b.k + bk - blockHalf;
                tap.weight = b.w * invNorm;
                kernel.taps.push_back(tap);
              }
            }
          }
        }

        // Base taps for the separable implementation: 13 entries in coarse-grid
        // units at positions (s * b.i, s * b.j, s * b.k) with weights b.w / 24.
        // Paired with an s^3 box-average of the input field this reproduces the
        // full kron(B_3D, J_s^3) / (24 s^3) stencil.
        for (const BaseEntry& b : base)
        {
          KernelTap tap;
          tap.dx = scale * b.i;
          tap.dy = scale * b.j;
          tap.dz = scale * b.k;
          tap.weight = b.w / 24.0;
          kernel.baseTaps.push_back(tap);
        }

        return kernel;
      }

      bool KernelScaleAwareQoiCalculator::IsInBounds(int x, int y, int z) const
      {
        const util::Vector3D<site_t>& dimensions = latticeData.GetSiteDimensions();
        return x >= 0 && y >= 0 && z >= 0 && x < static_cast<int>(dimensions.x) && y < static_cast<int>(dimensions.y)
        && z < static_cast<int>(dimensions.z);
      }

      bool KernelScaleAwareQoiCalculator::TryGetLocalIndex(int x, int y, int z, site_t& localIndex) const
      {
        if (!IsInBounds(x, y, z))
        {
          return false;
        }
        const util::Vector3D<site_t> sample(static_cast<site_t>(x), static_cast<site_t>(y), static_cast<site_t>(z));
        const site_t sampleGlobalId = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(sample);
        std::unordered_map<site_t, site_t>::const_iterator found = globalToLocal.find(sampleGlobalId);
        if (found == globalToLocal.end())
        {
          return false;
        }
        localIndex = found->second;
        return true;
      }

      void KernelScaleAwareQoiCalculator::BuildCoarseSiteIndices()
      {
        const site_t siteCount = latticeData.GetLocalFluidSiteCount();
        coarseSiteIndices.clear();
        coarseSiteIndices.reserve(siteCount);
        localIndexToCoarseIndex.assign(static_cast<std::size_t>(siteCount),
                                       static_cast<std::size_t>(-1));

        for (site_t i = 0; i < siteCount; ++i)
        {
          const util::Vector3D<site_t>& c = localGlobalCoords[i];
          if (coarseningFactor == 1 ||
              (c.x % coarseningFactor == 0 && c.y % coarseningFactor == 0 && c.z % coarseningFactor == 0))
          {
            localIndexToCoarseIndex[static_cast<std::size_t>(i)] = coarseSiteIndices.size();
            coarseSiteIndices.push_back(i);
          }
        }
      }

      KernelScaleAwareQoiCalculator::SamplePointRef
      KernelScaleAwareQoiCalculator::MakeSamplePointRef(int x, int y, int z) const
      {
        SamplePointRef ref;
        ref.x = x;
        ref.y = y;
        ref.z = z;
        ref.inBounds = IsInBounds(x, y, z);
        ref.isLocal = false;
        ref.localIndex = 0;
        ref.globalId = 0;

        if (!ref.inBounds)
        {
          return ref;
        }

        site_t localIndex;
        if (TryGetLocalIndex(x, y, z, localIndex))
        {
          ref.isLocal = true;
          ref.localIndex = localIndex;
          return ref;
        }

        ref.globalId = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
            util::Vector3D<site_t>(static_cast<site_t>(x), static_cast<site_t>(y), static_cast<site_t>(z)));
        return ref;
      }

      void KernelScaleAwareQoiCalculator::BuildEnstrophyNeighborRefs()
      {
        const int coarseStep = coarseningFactor;
        enstrophyNeighborRefs.resize(coarseSiteIndices.size());

        for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
        {
          const site_t i = coarseSiteIndices[siteIndex];
          const util::Vector3D<site_t>& center = localGlobalCoords[i];
          const int cx = static_cast<int>(center.x);
          const int cy = static_cast<int>(center.y);
          const int cz = static_cast<int>(center.z);

          enstrophyNeighborRefs[siteIndex][0] = MakeSamplePointRef(cx + coarseStep, cy, cz);
          enstrophyNeighborRefs[siteIndex][1] = MakeSamplePointRef(cx - coarseStep, cy, cz);
          enstrophyNeighborRefs[siteIndex][2] = MakeSamplePointRef(cx, cy + coarseStep, cz);
          enstrophyNeighborRefs[siteIndex][3] = MakeSamplePointRef(cx, cy - coarseStep, cz);
          enstrophyNeighborRefs[siteIndex][4] = MakeSamplePointRef(cx, cy, cz + coarseStep);
          enstrophyNeighborRefs[siteIndex][5] = MakeSamplePointRef(cx, cy, cz - coarseStep);
        }
      }

      util::Vector3D<distribn_t> KernelScaleAwareQoiCalculator::GetCoarseVelocity(const MacroscopicPropertyCache& propertyCache,
                                                                                   const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                                                                   int fineX, int fineY, int fineZ) const
      {
        if (coarseningFactor == 1)
        {
          // No coarse-graining: sample directly at fine lattice point.
          site_t idx;
          if (TryGetLocalIndex(fineX, fineY, fineZ, idx))
          {
            return propertyCache.postCollisionVelocityCache.Get(idx);
          }

          if (!IsInBounds(fineX, fineY, fineZ))
          {
            return ZeroVector();
          }

          const site_t globalId = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
              util::Vector3D<site_t>(static_cast<site_t>(fineX), static_cast<site_t>(fineY), static_cast<site_t>(fineZ)));
          std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator halo = haloVelocityCache.find(globalId);
          if (halo == haloVelocityCache.end())
          {
            return ZeroVector();
          }
          return halo->second;
        }

        // Coarse-graining: subsample one representative point from the m×m×m block
        // Coarse-grid block indices for this fine point
        int coarseX = fineX / coarseningFactor;
        int coarseY = fineY / coarseningFactor;
        int coarseZ = fineZ / coarseningFactor;

        // Fine-lattice coordinates of the coarse block corner
        int blockStartX = coarseX * coarseningFactor;
        int blockStartY = coarseY * coarseningFactor;
        int blockStartZ = coarseZ * coarseningFactor;

        // Subsample the block by taking the velocity at the block corner.
        site_t idx;
        if (TryGetLocalIndex(blockStartX, blockStartY, blockStartZ, idx))
        {
          return propertyCache.postCollisionVelocityCache.Get(idx);
        }

        if (!IsInBounds(blockStartX, blockStartY, blockStartZ))
        {
          return ZeroVector();
        }

        const site_t globalId = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
            util::Vector3D<site_t>(static_cast<site_t>(blockStartX), static_cast<site_t>(blockStartY), static_cast<site_t>(blockStartZ)));
        std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator halo = haloVelocityCache.find(globalId);
        if (halo == haloVelocityCache.end())
        {
          return ZeroVector();
        }
        return halo->second;
      }

      void KernelScaleAwareQoiCalculator::ComputeLocalIntegrals(const MacroscopicPropertyCache& propertyCache,
                                                                const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                                                const std::vector<KernelDefinition>& kernels,
                                                                std::vector<double>& localEnergy,
                                                                std::vector<double>& localEnstrophy) const
      {
        const site_t siteCount = latticeData.GetLocalFluidSiteCount();
        const std::size_t kernelCount = kernels.size();

        localEnergy.assign(kernelCount, 0.0);
        localEnstrophy.assign(kernelCount, 0.0);

        if (siteCount == 0 || kernelCount == 0)
        {
          return;
        }

        if (coarseSiteIndices.empty())
        {
          return;
        }

        if (filteredWorkspace.size() != kernelCount * siteCount)
        {
          filteredWorkspace.resize(kernelCount * siteCount, ZeroVector());
        }
        if (tempXWorkspace.size() != siteCount)
        {
          tempXWorkspace.resize(siteCount, ZeroVector());
        }
        if (tempYWorkspace.size() != siteCount)
        {
          tempYWorkspace.resize(siteCount, ZeroVector());
        }
        if (avgWorkspace.size() != siteCount)
        {
          avgWorkspace.resize(siteCount, ZeroVector());
        }
        if (remoteFilteredCache.size() != kernelCount)
        {
          remoteFilteredCache.resize(kernelCount);
        }

        const int coarseStep = coarseningFactor;

        for (std::size_t k = 0; k < kernelCount; ++k)
        {
          if (kernels[k].type == GaussianKernel)
          {
            const int radius = kernels[k].radius;
            const double s = static_cast<double>(kernels[k].scale);
            std::vector<double> w(2 * radius + 1, 0.0);
            double norm = 0.0;
            for (int d = -radius; d <= radius; ++d)
            {
              const double wd = std::exp(-(d * d) / (2.0 * s * s));
              w[d + radius] = wd;
              norm += wd;
            }
            for (int i = 0; i < 2 * radius + 1; ++i)
            {
              w[i] /= norm;
            }

            remoteTempXCache.clear();
            remoteTempYCache.clear();
            remoteTempXCache.reserve(coarseSiteIndices.size());
            remoteTempYCache.reserve(coarseSiteIndices.size());

            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dx = -radius; dx <= radius; ++dx)
              {
                sum = AddScaled(sum, GetCoarseVelocity(propertyCache,
                                                       haloVelocityCache,
                                                       static_cast<int>(c.x) + dx * coarseStep,
                                                       static_cast<int>(c.y),
                                                       static_cast<int>(c.z)),
                                w[dx + radius]);
              }
              tempXWorkspace[i] = sum;
            }

            const auto getTempXAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }

              if (ref.isLocal)
              {
                return tempXWorkspace[ref.localIndex];
              }

              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator found = remoteTempXCache.find(ref.globalId);
              if (found != remoteTempXCache.end())
              {
                return found->second;
              }

              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dx = -radius; dx <= radius; ++dx)
              {
                sum = AddScaled(sum, GetCoarseVelocity(propertyCache,
                                                       haloVelocityCache,
                                                       x + dx * coarseStep,
                                                       y,
                                                       z),
                                w[dx + radius]);
              }
              remoteTempXCache[ref.globalId] = sum;
              return sum;
            };

            const auto getTempYAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }

              if (ref.isLocal)
              {
                return tempYWorkspace[ref.localIndex];
              }

              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator found = remoteTempYCache.find(ref.globalId);
              if (found != remoteTempYCache.end())
              {
                return found->second;
              }

              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dy = -radius; dy <= radius; ++dy)
              {
                sum = AddScaled(sum, getTempXAt(x, y + dy * coarseStep, z), w[dy + radius]);
              }
              remoteTempYCache[ref.globalId] = sum;
              return sum;
            };

            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dy = -radius; dy <= radius; ++dy)
              {
                sum = AddScaled(sum,
                                getTempXAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y) + dy * coarseStep,
                                           static_cast<int>(c.z)),
                                w[dy + radius]);
              }
              tempYWorkspace[i] = sum;
            }

            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dz = -radius; dz <= radius; ++dz)
              {
                sum = AddScaled(sum,
                                getTempYAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y),
                                           static_cast<int>(c.z) + dz * coarseStep),
                                w[dz + radius]);
              }
              filteredWorkspace[k * siteCount + i] = sum;
            }
          }
          else
          {
            // Laplacian kernel: apply L_s = kron(B_3D, J_s^3) / (24 s^3) as three
            // 1D box-average passes followed by a 13-tap base stencil.
            //   Pass 1 (x): tempX(x,y,z) = (1/s) sum_{dx} v(x+dx, y, z)
            //   Pass 2 (y): tempY(x,y,z) = (1/s) sum_{dy} tempX(x, y+dy, z)
            //   Pass 3 (z): avg  (x,y,z) = (1/s) sum_{dz} tempY(x, y, z+dz)
            //   Final:      L_s*v(x)    = sum_{b in baseTaps} b.weight * avg(x + b)
            // The box-avg offsets run over dd ∈ {-blockHalf..+blockHalf} * coarseStep,
            // where blockHalf = (s-1)/2. For s = 1 each box-avg is a no-op.
            const int s = kernels[k].scale;
            const int blockHalf = (s - 1) / 2;
            const double invS = 1.0 / static_cast<double>(s);

            remoteTempXCache.clear();
            remoteTempYCache.clear();
            remoteAvgCache.clear();
            remoteTempXCache.reserve(coarseSiteIndices.size());
            remoteTempYCache.reserve(coarseSiteIndices.size());
            remoteAvgCache.reserve(coarseSiteIndices.size());

            // Pass 1: tempX along x (raw velocity input).
            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, GetCoarseVelocity(propertyCache,
                                                       haloVelocityCache,
                                                       static_cast<int>(c.x) + dd * coarseStep,
                                                       static_cast<int>(c.y),
                                                       static_cast<int>(c.z)),
                                invS);
              }
              tempXWorkspace[i] = sum;
            }

            const auto getTempXAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempXWorkspace[ref.localIndex];
              }
              auto found = remoteTempXCache.find(ref.globalId);
              if (found != remoteTempXCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, GetCoarseVelocity(propertyCache,
                                                       haloVelocityCache,
                                                       x + dd * coarseStep,
                                                       y,
                                                       z),
                                invS);
              }
              remoteTempXCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 2: tempY along y (from tempX).
            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum,
                                getTempXAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y) + dd * coarseStep,
                                           static_cast<int>(c.z)),
                                invS);
              }
              tempYWorkspace[i] = sum;
            }

            const auto getTempYAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempYWorkspace[ref.localIndex];
              }
              auto found = remoteTempYCache.find(ref.globalId);
              if (found != remoteTempYCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, getTempXAt(x, y + dd * coarseStep, z), invS);
              }
              remoteTempYCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 3: avgWorkspace along z (from tempY). Result is box_s^3 * v.
            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum,
                                getTempYAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y),
                                           static_cast<int>(c.z) + dd * coarseStep),
                                invS);
              }
              avgWorkspace[i] = sum;
            }

            const auto getAvgAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return avgWorkspace[ref.localIndex];
              }
              auto found = remoteAvgCache.find(ref.globalId);
              if (found != remoteAvgCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, getTempYAt(x, y, z + dd * coarseStep), invS);
              }
              remoteAvgCache[ref.globalId] = sum;
              return sum;
            };

            // Final: apply the 13-tap base stencil to the box-averaged field.
            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (const KernelTap& tap : kernels[k].baseTaps)
              {
                sum = AddScaled(sum,
                                getAvgAt(static_cast<int>(c.x) + tap.dx * coarseStep,
                                         static_cast<int>(c.y) + tap.dy * coarseStep,
                                         static_cast<int>(c.z) + tap.dz * coarseStep),
                                tap.weight);
              }
              filteredWorkspace[k * siteCount + i] = sum;
            }
          }
        }

        const double invTwo = 0.5;
        const double invTwoCoarseStep = invTwo / static_cast<double>(coarseningFactor);

        for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
        {
          const site_t i = coarseSiteIndices[siteIndex];

          for (std::size_t k = 0; k < kernelCount; ++k)
          {
            if (siteIndex == 0)
            {
              remoteFilteredCache[k].clear();
              remoteFilteredCache[k].reserve(coarseSiteIndices.size());
            }

            const util::Vector3D<distribn_t>& vf = filteredWorkspace[k * siteCount + i];
            localEnergy[k] += invTwo * (vf.x * vf.x + vf.y * vf.y + vf.z * vf.z);

            const auto sampleFiltered = [&](const SamplePointRef& ref) -> util::Vector3D<distribn_t>
            {
              if (!ref.inBounds)
              {
                return ZeroVector();
              }

              if (ref.isLocal)
              {
                return filteredWorkspace[k * siteCount + ref.localIndex];
              }

              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator cached = remoteFilteredCache[k].find(ref.globalId);
              if (cached != remoteFilteredCache[k].end())
              {
                return cached->second;
              }

              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              const int sx = ref.x;
              const int sy = ref.y;
              const int sz = ref.z;
              if (kernels[k].type == GaussianKernel)
              {
                for (std::size_t t = 0; t < kernels[k].taps.size(); ++t)
                {
                  const KernelTap& tap = kernels[k].taps[t];
                  const util::Vector3D<distribn_t>& velocity = GetCoarseVelocity(propertyCache,
                                                                                   haloVelocityCache,
                                                                                   sx + tap.dx * coarseStep,
                                                                                   sy + tap.dy * coarseStep,
                                                                                   sz + tap.dz * coarseStep);
                  sum.x += tap.weight * velocity.x;
                  sum.y += tap.weight * velocity.y;
                  sum.z += tap.weight * velocity.z;
                }
              }
              else
              {
                for (std::size_t t = 0; t < kernels[k].taps.size(); ++t)
                {
                  const KernelTap& tap = kernels[k].taps[t];
                  const util::Vector3D<distribn_t>& velocity = GetCoarseVelocity(propertyCache,
                                                                                   haloVelocityCache,
                                                                                   sx + tap.dx * coarseStep,
                                                                                   sy + tap.dy * coarseStep,
                                                                                   sz + tap.dz * coarseStep);
                  sum.x += tap.weight * velocity.x;
                  sum.y += tap.weight * velocity.y;
                  sum.z += tap.weight * velocity.z;
                }
              }

              remoteFilteredCache[k][ref.globalId] = sum;
              return sum;
            };

            const std::array<SamplePointRef, 6>& refs = enstrophyNeighborRefs[siteIndex];
            const util::Vector3D<distribn_t> xp = sampleFiltered(refs[0]);
            const util::Vector3D<distribn_t> xm = sampleFiltered(refs[1]);
            const util::Vector3D<distribn_t> yp = sampleFiltered(refs[2]);
            const util::Vector3D<distribn_t> ym = sampleFiltered(refs[3]);
            const util::Vector3D<distribn_t> zp = sampleFiltered(refs[4]);
            const util::Vector3D<distribn_t> zm = sampleFiltered(refs[5]);

            const double dVxDy = invTwoCoarseStep * (yp.x - ym.x);
            const double dVxDz = invTwoCoarseStep * (zp.x - zm.x);
            const double dVyDx = invTwoCoarseStep * (xp.y - xm.y);
            const double dVyDz = invTwoCoarseStep * (zp.y - zm.y);
            const double dVzDx = invTwoCoarseStep * (xp.z - xm.z);
            const double dVzDy = invTwoCoarseStep * (yp.z - ym.z);

            const double wx = dVzDy - dVyDz;
            const double wy = dVxDz - dVzDx;
            const double wz = dVyDx - dVxDy;

            localEnstrophy[k] += invTwo * (wx * wx + wy * wy + wz * wz);
          }
        }
      }
      util::Vector3D<distribn_t> KernelScaleAwareQoiCalculator::SampleFiltered(int x, int y, int z,
                                                                               std::size_t k) const
      {
        if (!IsInBounds(x, y, z))
        {
          return ZeroVector();
        }
        const site_t siteCount = latticeData.GetLocalFluidSiteCount();
        site_t localIdx;
        if (TryGetLocalIndex(x, y, z, localIdx))
        {
          return filteredWorkspace[k * siteCount + localIdx];
        }
        const site_t gid = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
            util::Vector3D<site_t>(static_cast<site_t>(x), static_cast<site_t>(y), static_cast<site_t>(z)));
        if (k < remoteFilteredCache.size())
        {
          auto it = remoteFilteredCache[k].find(gid);
          if (it != remoteFilteredCache[k].end())
          {
            return it->second;
          }
        }
        return ZeroVector();
      }

      void KernelScaleAwareQoiCalculator::ComputeLocalFDFieldsAndGram(
          const MacroscopicPropertyCache& propertyCache,
          const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
          const std::vector<KernelDefinition>& kernels,
          std::vector<std::vector<util::Vector3D<distribn_t> > >& energyFDs,
          std::vector<std::vector<util::Vector3D<distribn_t> > >& enstrophyFDs,
          std::vector<double>& gramMatrix) const
      {
        // This method must be called after ComputeLocalIntegrals so that filteredWorkspace
        // and remoteFilteredCache are populated.
        const std::size_t nKernels = kernels.size();
        const std::size_t nCoarse = coarseSiteIndices.size();
        const std::size_t nQoi = 2 * nKernels; // energy QoIs then enstrophy QoIs

        energyFDs.assign(nKernels, std::vector<util::Vector3D<distribn_t> >(nCoarse, ZeroVector()));
        enstrophyFDs.assign(nKernels, std::vector<util::Vector3D<distribn_t> >(nCoarse, ZeroVector()));
        gramMatrix.assign(nQoi * nQoi, 0.0);

        if (nCoarse == 0 || nKernels == 0)
        {
          return;
        }

        const int coarseStep = coarseningFactor;
        const double h = static_cast<double>(coarseStep);
        const double invH2 = 1.0 / (h * h);

        // Steps 1 & 2 are merged per-kernel so the separable caches populated while
        // building K*K*v at local coarse sites can be reused when Step 2 needs K*K*v
        // at cross-rank neighbors. This turns the Gaussian FD step from O(radius^3)
        // to O(radius) per tap and avoids any 15,625-tap enumeration.
        for (std::size_t k = 0; k < nKernels; ++k)
        {
          // Per-kernel remote caches (cleared before each pass, populated on demand).
          remoteTempXCache.clear();
          remoteTempYCache.clear();
          remoteTempXCache.reserve(nCoarse);
          remoteTempYCache.reserve(nCoarse);

          if (kernels[k].type == GaussianKernel)
          {
            // Apply K separably to the single-filtered field K*v (already cached in
            // filteredWorkspace / remoteFilteredCache from ComputeLocalIntegrals),
            // producing K*K*v by three 1D Gaussian passes in x, y, z.
            const int radius = kernels[k].radius;
            const double s = static_cast<double>(kernels[k].scale);
            std::vector<double> w(2 * radius + 1, 0.0);
            double norm = 0.0;
            for (int d = -radius; d <= radius; ++d)
            {
              const double wd = std::exp(-(d * d) / (2.0 * s * s));
              w[d + radius] = wd;
              norm += wd;
            }
            for (int i = 0; i < 2 * radius + 1; ++i)
            {
              w[i] /= norm;
            }

            // Pass 1 (x) of K*v -> tempX.
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dx = -radius; dx <= radius; ++dx)
              {
                sum = AddScaled(sum,
                                SampleFiltered(static_cast<int>(c.x) + dx * coarseStep,
                                               static_cast<int>(c.y),
                                               static_cast<int>(c.z),
                                               k),
                                w[dx + radius]);
              }
              tempXWorkspace[i] = sum;
            }

            const auto getTempXAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempXWorkspace[ref.localIndex];
              }
              auto found = remoteTempXCache.find(ref.globalId);
              if (found != remoteTempXCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dx = -radius; dx <= radius; ++dx)
              {
                sum = AddScaled(sum, SampleFiltered(x + dx * coarseStep, y, z, k), w[dx + radius]);
              }
              remoteTempXCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 2 (y) of tempX -> tempY.
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dy = -radius; dy <= radius; ++dy)
              {
                sum = AddScaled(sum,
                                getTempXAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y) + dy * coarseStep,
                                           static_cast<int>(c.z)),
                                w[dy + radius]);
              }
              tempYWorkspace[i] = sum;
            }

            const auto getTempYAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempYWorkspace[ref.localIndex];
              }
              auto found = remoteTempYCache.find(ref.globalId);
              if (found != remoteTempYCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dy = -radius; dy <= radius; ++dy)
              {
                sum = AddScaled(sum, getTempXAt(x, y + dy * coarseStep, z), w[dy + radius]);
              }
              remoteTempYCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 3 (z) of tempY -> energyFDs[k] (K*K*v at local coarse sites).
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dz = -radius; dz <= radius; ++dz)
              {
                sum = AddScaled(sum,
                                getTempYAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y),
                                           static_cast<int>(c.z) + dz * coarseStep),
                                w[dz + radius]);
              }
              energyFDs[k][siteIndex] = sum;
            }

            // Step 2 (this kernel): enstrophy FD = -Laplacian_h(K*K*v).
            std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteEnergyFDCache;
            const auto remoteEnergyFD = [&](int rx, int ry, int rz) -> util::Vector3D<distribn_t>
            {
              // K*K*v at a remote in-bounds coarse site = one more 1D z-pass
              // over getTempYAt (O(2*radius+1), not O(radius^3)).
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dz = -radius; dz <= radius; ++dz)
              {
                sum = AddScaled(sum, getTempYAt(rx, ry, rz + dz * coarseStep), w[dz + radius]);
              }
              return sum;
            };

            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const util::Vector3D<distribn_t>& wc = energyFDs[k][siteIndex];
              util::Vector3D<distribn_t> laplacian(6.0 * invH2 * wc.x,
                                                   6.0 * invH2 * wc.y,
                                                   6.0 * invH2 * wc.z);
              const std::array<SamplePointRef, 6>& refs = enstrophyNeighborRefs[siteIndex];
              for (int dir = 0; dir < 6; ++dir)
              {
                const SamplePointRef& ref = refs[dir];
                if (!ref.inBounds)
                {
                  continue;
                }
                util::Vector3D<distribn_t> nbr(0.0, 0.0, 0.0);
                if (ref.isLocal)
                {
                  const std::size_t ni = localIndexToCoarseIndex[static_cast<std::size_t>(ref.localIndex)];
                  if (ni != static_cast<std::size_t>(-1))
                  {
                    nbr = energyFDs[k][ni];
                  }
                }
                else
                {
                  auto it = remoteEnergyFDCache.find(ref.globalId);
                  if (it != remoteEnergyFDCache.end())
                  {
                    nbr = it->second;
                  }
                  else
                  {
                    nbr = remoteEnergyFD(ref.x, ref.y, ref.z);
                    remoteEnergyFDCache[ref.globalId] = nbr;
                  }
                }
                laplacian.x -= invH2 * nbr.x;
                laplacian.y -= invH2 * nbr.y;
                laplacian.z -= invH2 * nbr.z;
              }
              enstrophyFDs[k][siteIndex] = laplacian;
            }
          }
          else
          {
            // Laplacian kernel: K*K*v = L_s*(L_s*v) applied separably (3 box-avg
            // passes of L_s*v, then 13-tap base stencil), mirroring ComputeLocalIntegrals.
            const int s = kernels[k].scale;
            const int blockHalf = (s - 1) / 2;
            const double invS = 1.0 / static_cast<double>(s);

            remoteAvgCache.clear();
            remoteAvgCache.reserve(nCoarse);

            // Pass 1: tempX along x of filteredWorkspace (which stores L_s*v).
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum,
                                SampleFiltered(static_cast<int>(c.x) + dd * coarseStep,
                                               static_cast<int>(c.y),
                                               static_cast<int>(c.z),
                                               k),
                                invS);
              }
              tempXWorkspace[i] = sum;
            }

            const auto getTempXAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempXWorkspace[ref.localIndex];
              }
              auto found = remoteTempXCache.find(ref.globalId);
              if (found != remoteTempXCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, SampleFiltered(x + dd * coarseStep, y, z, k), invS);
              }
              remoteTempXCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 2: tempY along y from tempX.
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum,
                                getTempXAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y) + dd * coarseStep,
                                           static_cast<int>(c.z)),
                                invS);
              }
              tempYWorkspace[i] = sum;
            }

            const auto getTempYAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return tempYWorkspace[ref.localIndex];
              }
              auto found = remoteTempYCache.find(ref.globalId);
              if (found != remoteTempYCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, getTempXAt(x, y + dd * coarseStep, z), invS);
              }
              remoteTempYCache[ref.globalId] = sum;
              return sum;
            };

            // Pass 3: avg along z from tempY.
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum,
                                getTempYAt(static_cast<int>(c.x),
                                           static_cast<int>(c.y),
                                           static_cast<int>(c.z) + dd * coarseStep),
                                invS);
              }
              avgWorkspace[i] = sum;
            }

            const auto getAvgAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              const SamplePointRef ref = MakeSamplePointRef(x, y, z);
              if (!ref.inBounds)
              {
                return ZeroVector();
              }
              if (ref.isLocal)
              {
                return avgWorkspace[ref.localIndex];
              }
              auto found = remoteAvgCache.find(ref.globalId);
              if (found != remoteAvgCache.end())
              {
                return found->second;
              }
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dd = -blockHalf; dd <= blockHalf; ++dd)
              {
                sum = AddScaled(sum, getTempYAt(x, y, z + dd * coarseStep), invS);
              }
              remoteAvgCache[ref.globalId] = sum;
              return sum;
            };

            // Apply the 13-tap base stencil to the box-averaged L_s*v, producing L_s*L_s*v.
            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& c = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (const KernelTap& tap : kernels[k].baseTaps)
              {
                sum = AddScaled(sum,
                                getAvgAt(static_cast<int>(c.x) + tap.dx * coarseStep,
                                         static_cast<int>(c.y) + tap.dy * coarseStep,
                                         static_cast<int>(c.z) + tap.dz * coarseStep),
                                tap.weight);
              }
              energyFDs[k][siteIndex] = sum;
            }

            // Step 2 (this kernel): same structure — at a remote in-bounds coarse
            // neighbor K*K*v is the 13-tap base-stencil application on getAvgAt.
            std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteEnergyFDCache;
            const auto remoteEnergyFD = [&](int rx, int ry, int rz) -> util::Vector3D<distribn_t>
            {
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (const KernelTap& tap : kernels[k].baseTaps)
              {
                sum = AddScaled(sum,
                                getAvgAt(rx + tap.dx * coarseStep,
                                         ry + tap.dy * coarseStep,
                                         rz + tap.dz * coarseStep),
                                tap.weight);
              }
              return sum;
            };

            for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
            {
              const util::Vector3D<distribn_t>& wc = energyFDs[k][siteIndex];
              util::Vector3D<distribn_t> laplacian(6.0 * invH2 * wc.x,
                                                   6.0 * invH2 * wc.y,
                                                   6.0 * invH2 * wc.z);
              const std::array<SamplePointRef, 6>& refs = enstrophyNeighborRefs[siteIndex];
              for (int dir = 0; dir < 6; ++dir)
              {
                const SamplePointRef& ref = refs[dir];
                if (!ref.inBounds)
                {
                  continue;
                }
                util::Vector3D<distribn_t> nbr(0.0, 0.0, 0.0);
                if (ref.isLocal)
                {
                  const std::size_t ni = localIndexToCoarseIndex[static_cast<std::size_t>(ref.localIndex)];
                  if (ni != static_cast<std::size_t>(-1))
                  {
                    nbr = energyFDs[k][ni];
                  }
                }
                else
                {
                  auto it = remoteEnergyFDCache.find(ref.globalId);
                  if (it != remoteEnergyFDCache.end())
                  {
                    nbr = it->second;
                  }
                  else
                  {
                    nbr = remoteEnergyFD(ref.x, ref.y, ref.z);
                    remoteEnergyFDCache[ref.globalId] = nbr;
                  }
                }
                laplacian.x -= invH2 * nbr.x;
                laplacian.y -= invH2 * nbr.y;
                laplacian.z -= invH2 * nbr.z;
              }
              enstrophyFDs[k][siteIndex] = laplacian;
            }
          }
        }

        // Step 3: compute Gram matrix local contributions G[i,j] = sum_x V_i(x) . V_j(x).
        // QoI ordering: energies first (indices 0..nKernels-1), then enstrophies (nKernels..2*nKernels-1).
        for (std::size_t siteIndex = 0; siteIndex < nCoarse; ++siteIndex)
        {
          for (std::size_t ki = 0; ki < nKernels; ++ki)
          {
            const util::Vector3D<distribn_t>& Ve = energyFDs[ki][siteIndex];
            const util::Vector3D<distribn_t>& Vz = enstrophyFDs[ki][siteIndex];

            for (std::size_t kj = 0; kj < nKernels; ++kj)
            {
              const util::Vector3D<distribn_t>& We = energyFDs[kj][siteIndex];
              const util::Vector3D<distribn_t>& Wz = enstrophyFDs[kj][siteIndex];

              const std::size_t iE = ki;
              const std::size_t iZ = ki + nKernels;
              const std::size_t jE = kj;
              const std::size_t jZ = kj + nKernels;

              // <Ve_ki, Ve_kj>
              gramMatrix[iE * nQoi + jE] += Ve.x * We.x + Ve.y * We.y + Ve.z * We.z;
              // <Ve_ki, Vz_kj>
              gramMatrix[iE * nQoi + jZ] += Ve.x * Wz.x + Ve.y * Wz.y + Ve.z * Wz.z;
              // <Vz_ki, Ve_kj>
              gramMatrix[iZ * nQoi + jE] += Vz.x * We.x + Vz.y * We.y + Vz.z * We.z;
              // <Vz_ki, Vz_kj>
              gramMatrix[iZ * nQoi + jZ] += Vz.x * Wz.x + Vz.y * Wz.y + Vz.z * Wz.z;
            }
          }
        }
      }

      std::vector<util::Vector3D<distribn_t> >
      KernelScaleAwareQoiCalculator::BuildPerSiteDeltaV(
          const std::vector<util::Vector3D<distribn_t> >& perCoarseSiteDeltaV) const
      {
        const site_t siteCount = latticeData.GetLocalFluidSiteCount();
        std::vector<util::Vector3D<distribn_t> > result(siteCount, ZeroVector());
        for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
        {
          result[coarseSiteIndices[siteIndex]] = perCoarseSiteDeltaV[siteIndex];
        }
        return result;
      }

    }
  }
}

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
        KernelDefinition kernel;
        kernel.label = label;
        kernel.type = LaplacianKernel;
        kernel.scale = scale;
        kernel.radius = scale;

        const double s = static_cast<double>(scale);
        const double centerWeight = 6.0 / (s * s);
        const double axisWeight = -1.0 / (s * s);

        kernel.taps.push_back(KernelTap{0, 0, 0, centerWeight});
        kernel.taps.push_back(KernelTap{scale, 0, 0, axisWeight});
        kernel.taps.push_back(KernelTap{-scale, 0, 0, axisWeight});
        kernel.taps.push_back(KernelTap{0, scale, 0, axisWeight});
        kernel.taps.push_back(KernelTap{0, -scale, 0, axisWeight});
        kernel.taps.push_back(KernelTap{0, 0, scale, axisWeight});
        kernel.taps.push_back(KernelTap{0, 0, -scale, axisWeight});

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

        std::vector<util::Vector3D<distribn_t> > filtered(kernelCount * siteCount, ZeroVector());
        std::vector<site_t> coarseSiteIndices;
        coarseSiteIndices.reserve(siteCount);

        for (site_t i = 0; i < siteCount; ++i)
        {
          const util::Vector3D<site_t>& c = localGlobalCoords[i];
          if (coarseningFactor == 1 ||
              (c.x % coarseningFactor == 0 && c.y % coarseningFactor == 0 && c.z % coarseningFactor == 0))
          {
            coarseSiteIndices.push_back(i);
          }
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

            std::vector<util::Vector3D<distribn_t> > tempX(siteCount, ZeroVector());
            std::vector<util::Vector3D<distribn_t> > tempY(siteCount, ZeroVector());
            std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteTempX;
            std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteTempY;

            const auto getGlobalId = [&](int x, int y, int z) -> site_t
            {
              return latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
                  util::Vector3D<site_t>(static_cast<site_t>(x), static_cast<site_t>(y), static_cast<site_t>(z)));
            };

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
              tempX[i] = sum;
            }

            const auto getTempXAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              if (!IsInBounds(x, y, z))
              {
                return ZeroVector();
              }

              site_t idx;
              if (TryGetLocalIndex(x, y, z, idx))
              {
                return tempX[idx];
              }

              const site_t gid = getGlobalId(x, y, z);
              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator found = remoteTempX.find(gid);
              if (found != remoteTempX.end())
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
              remoteTempX[gid] = sum;
              return sum;
            };

            const auto getTempYAt = [&](int x, int y, int z) -> util::Vector3D<distribn_t>
            {
              if (!IsInBounds(x, y, z))
              {
                return ZeroVector();
              }

              site_t idx;
              if (TryGetLocalIndex(x, y, z, idx))
              {
                return tempY[idx];
              }

              const site_t gid = getGlobalId(x, y, z);
              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator found = remoteTempY.find(gid);
              if (found != remoteTempY.end())
              {
                return found->second;
              }

              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
              for (int dy = -radius; dy <= radius; ++dy)
              {
                sum = AddScaled(sum, getTempXAt(x, y + dy * coarseStep, z), w[dy + radius]);
              }
              remoteTempY[gid] = sum;
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
              tempY[i] = sum;
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
              filtered[k * siteCount + i] = sum;
            }
          }
          else
          {
            for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
            {
              const site_t i = coarseSiteIndices[siteIndex];
              const util::Vector3D<site_t>& center = localGlobalCoords[i];
              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);

              for (std::size_t t = 0; t < kernels[k].taps.size(); ++t)
              {
                const KernelTap& tap = kernels[k].taps[t];
                const util::Vector3D<distribn_t>& velocity = GetCoarseVelocity(propertyCache,
                                                                                 haloVelocityCache,
                                                                                 static_cast<int>(center.x) + tap.dx * coarseStep,
                                                                                 static_cast<int>(center.y) + tap.dy * coarseStep,
                                                                                 static_cast<int>(center.z) + tap.dz * coarseStep);
                sum.x += tap.weight * velocity.x;
                sum.y += tap.weight * velocity.y;
                sum.z += tap.weight * velocity.z;
              }

              filtered[k * siteCount + i] = sum;
            }
          }
        }

        const double invTwo = 0.5;
        const double invTwoCoarseStep = invTwo / static_cast<double>(coarseningFactor);
        std::vector<std::unordered_map<site_t, util::Vector3D<distribn_t> > > remoteFiltered(kernelCount);

        for (std::size_t siteIndex = 0; siteIndex < coarseSiteIndices.size(); ++siteIndex)
        {
          const site_t i = coarseSiteIndices[siteIndex];
          const util::Vector3D<site_t>& center = localGlobalCoords[i];

          for (std::size_t k = 0; k < kernelCount; ++k)
          {
            const util::Vector3D<distribn_t>& vf = filtered[k * siteCount + i];
            localEnergy[k] += invTwo * (vf.x * vf.x + vf.y * vf.y + vf.z * vf.z);

            const auto sampleFiltered = [&](int ox, int oy, int oz) -> util::Vector3D<distribn_t>
            {
              const int sx = static_cast<int>(center.x) + ox * coarseStep;
              const int sy = static_cast<int>(center.y) + oy * coarseStep;
              const int sz = static_cast<int>(center.z) + oz * coarseStep;

              if (!IsInBounds(sx, sy, sz))
              {
                return ZeroVector();
              }

              site_t idx;
              if (TryGetLocalIndex(sx, sy, sz, idx))
              {
                return filtered[k * siteCount + idx];
              }

              const site_t gid = latticeData.GetGlobalNoncontiguousSiteIdFromGlobalCoords(
                  util::Vector3D<site_t>(static_cast<site_t>(sx), static_cast<site_t>(sy), static_cast<site_t>(sz)));
              std::unordered_map<site_t, util::Vector3D<distribn_t> >::const_iterator cached = remoteFiltered[k].find(gid);
              if (cached != remoteFiltered[k].end())
              {
                return cached->second;
              }

              util::Vector3D<distribn_t> sum(0.0, 0.0, 0.0);
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

              remoteFiltered[k][gid] = sum;
              return sum;
            };

            const util::Vector3D<distribn_t> xp = sampleFiltered(1, 0, 0);
            const util::Vector3D<distribn_t> xm = sampleFiltered(-1, 0, 0);
            const util::Vector3D<distribn_t> yp = sampleFiltered(0, 1, 0);
            const util::Vector3D<distribn_t> ym = sampleFiltered(0, -1, 0);
            const util::Vector3D<distribn_t> zp = sampleFiltered(0, 0, 1);
            const util::Vector3D<distribn_t> zm = sampleFiltered(0, 0, -1);

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
    }
  }
}

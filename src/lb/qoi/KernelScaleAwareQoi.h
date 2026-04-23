// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H
#define HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H

#include <array>
#include <vector>
#include <unordered_map>
#include <string>

#include "geometry/LatticeData.h"
#include "lb/MacroscopicPropertyCache.h"

namespace hemelb
{
  namespace lb
  {
    namespace qoi
    {
      enum KernelType
      {
        GaussianKernel,
        LaplacianKernel
      };

      struct KernelTap
      {
        int dx;
        int dy;
        int dz;
        double weight;
      };

      struct KernelDefinition
      {
        std::string label;
        KernelType type;
        int scale;
        std::vector<KernelTap> taps;
        // For Laplacian kernels: the 13-entry star-shaped base stencil expressed
        // in coarse-grid units (tap offsets are multiples of `scale`, weights are
        // base_weight / 24). Used together with an s^3 box-average of the input
        // field to apply L_s as three 1D box-avg passes + a 13-tap application,
        // equivalent to the full kron(B_3D, J_s^3) / (24 s^3) stencil but much
        // cheaper for s > 1. Empty for Gaussian kernels.
        std::vector<KernelTap> baseTaps;
        int radius;
      };

      class KernelScaleAwareQoiCalculator
      {
        public:
          explicit KernelScaleAwareQoiCalculator(const geometry::LatticeData& latticeData,
                                                 int coarseningFactor = 1);

          std::vector<KernelDefinition> CreateDefaultKernels() const;

          std::size_t GetCoarseSiteCount() const
          {
            return coarseSiteIndices.size();
          }

          void ComputeLocalIntegrals(const MacroscopicPropertyCache& propertyCache,
                                     const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                     const std::vector<KernelDefinition>& kernels,
                                     std::vector<double>& localEnergy,
                                     std::vector<double>& localEnstrophy) const;

          // Must be called after ComputeLocalIntegrals (reuses filteredWorkspace and remoteFilteredCache).
          // energyFDs[k][siteIndex]    = K_k K_k v  (double-filtered velocity, functional deriv of E_{K_k})
          // enstrophyFDs[k][siteIndex] = -L_h(K_k K_k v) = discrete curl^*(curl(K_k K_k v)) via Laplacian approx
          // gramMatrix[i*nQoi + j]     = local contribution to <V_i, V_j> summed over coarse sites
          // QoI ordering: [E_k0, E_k1, ..., Z_k0, Z_k1, ...]  (nKernel energy then nKernel enstrophy)
          void ComputeLocalFDFieldsAndGram(
              const MacroscopicPropertyCache& propertyCache,
              const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
              const std::vector<KernelDefinition>& kernels,
              std::vector<std::vector<util::Vector3D<distribn_t> > >& energyFDs,
              std::vector<std::vector<util::Vector3D<distribn_t> > >& enstrophyFDs,
              std::vector<double>& gramMatrix) const;

          // Apply per-coarse-site velocity correction to all local fluid sites.
          // perCoarseSiteDeltaV[siteIndex] gives DeltaV for coarseSiteIndices[siteIndex].
          // Returns a per-local-site deltaV vector (size = localFluidSiteCount), zero for non-coarse sites.
          std::vector<util::Vector3D<distribn_t> > BuildPerSiteDeltaV(
              const std::vector<util::Vector3D<distribn_t> >& perCoarseSiteDeltaV) const;

        private:
          struct SamplePointRef
          {
            int x;
            int y;
            int z;
            bool inBounds;
            bool isLocal;
            site_t localIndex;
            site_t globalId;
          };

          static KernelDefinition CreateGaussianKernel(const std::string& label, int scale);
          static KernelDefinition CreateLaplacianKernel(const std::string& label, int scale);

          bool IsInBounds(int x, int y, int z) const;
          bool TryGetLocalIndex(int x, int y, int z, site_t& localIndex) const;
          void BuildCoarseSiteIndices();
          void BuildEnstrophyNeighborRefs();
          SamplePointRef MakeSamplePointRef(int x, int y, int z) const;
          
          // Coarse-graining helper: gets coarse-grained velocity by subsampling a fine-grid block
          util::Vector3D<distribn_t> GetCoarseVelocity(const MacroscopicPropertyCache& propertyCache,
                                                        const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                                        int fineX, int fineY, int fineZ) const;

          // Look up the single-filtered velocity K_k v at a coarse-grid point from cached workspace.
          // Falls back to zero-padding if the site is out-of-bounds or not in the cache.
          util::Vector3D<distribn_t> SampleFiltered(int x, int y, int z, std::size_t k) const;

          std::vector<util::Vector3D<site_t> > localGlobalCoords;
          std::unordered_map<site_t, site_t> globalToLocal;
          std::vector<site_t> coarseSiteIndices;
          // Inverse of coarseSiteIndices: for each local fluid index, the position in
          // coarseSiteIndices (or static_cast<std::size_t>(-1) if not a coarse site).
          // Enables O(1) local-neighbor lookups in ComputeLocalFDFieldsAndGram.
          std::vector<std::size_t> localIndexToCoarseIndex;
          std::vector<std::array<SamplePointRef, 6> > enstrophyNeighborRefs;
          mutable std::vector<util::Vector3D<distribn_t> > filteredWorkspace;
          mutable std::vector<util::Vector3D<distribn_t> > tempXWorkspace;
          mutable std::vector<util::Vector3D<distribn_t> > tempYWorkspace;
          mutable std::vector<util::Vector3D<distribn_t> > avgWorkspace;
          mutable std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteTempXCache;
          mutable std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteTempYCache;
          mutable std::unordered_map<site_t, util::Vector3D<distribn_t> > remoteAvgCache;
          mutable std::vector<std::unordered_map<site_t, util::Vector3D<distribn_t> > > remoteFilteredCache;

          const geometry::LatticeData& latticeData;
          int coarseningFactor;
      };
    }
  }
}

#endif // HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H

// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H
#define HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H

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
        int radius;
      };

      class KernelScaleAwareQoiCalculator
      {
        public:
          explicit KernelScaleAwareQoiCalculator(const geometry::LatticeData& latticeData,
                                                 int coarseningFactor = 1);

          std::vector<KernelDefinition> CreateDefaultKernels() const;

          void ComputeLocalIntegrals(const MacroscopicPropertyCache& propertyCache,
                                     const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                     const std::vector<KernelDefinition>& kernels,
                                     std::vector<double>& localEnergy,
                                     std::vector<double>& localEnstrophy) const;

        private:
          static KernelDefinition CreateGaussianKernel(const std::string& label, int scale);
          static KernelDefinition CreateLaplacianKernel(const std::string& label, int scale);

          bool IsInBounds(int x, int y, int z) const;
          bool TryGetLocalIndex(int x, int y, int z, site_t& localIndex) const;
          
          // Coarse-graining helper: gets coarse-grained velocity by subsampling a fine-grid block
          util::Vector3D<distribn_t> GetCoarseVelocity(const MacroscopicPropertyCache& propertyCache,
                                                        const std::unordered_map<site_t, util::Vector3D<distribn_t> >& haloVelocityCache,
                                                        int fineX, int fineY, int fineZ) const;

          std::vector<util::Vector3D<site_t> > localGlobalCoords;
          std::unordered_map<site_t, site_t> globalToLocal;

          const geometry::LatticeData& latticeData;
          int coarseningFactor;
      };
    }
  }
}

#endif // HEMELB_LB_QOI_KERNELSCALEAWAREQOI_H

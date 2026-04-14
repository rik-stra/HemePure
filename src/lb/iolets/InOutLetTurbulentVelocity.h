// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_IOLETS_INOUTLETTURBULENTVELOCITY_H
#define HEMELB_LB_IOLETS_INOUTLETTURBULENTVELOCITY_H

#include "lb/iolets/InOutLetVelocity.h"

namespace hemelb
{
  namespace lb
  {
    namespace iolets
    {
      class InOutLetTurbulentVelocity : public InOutLetVelocity
      {
        public:
          InOutLetTurbulentVelocity();
          virtual ~InOutLetTurbulentVelocity();
          InOutLet* Clone() const;

          LatticeVelocity GetVelocity(const LatticePosition& x, const LatticeTimeStep t) const;

          void SetMaxSpeed(const LatticeSpeed& v)
          {
            maxSpeed = v;
          }

          void SetIntensity(const Dimensionless& i)
          {
            intensity = i;
          }

          void SetWarmup(const LatticeTimeStep warmup)
          {
            warmUpLength = warmup;
          }

          void SetSeed(const LatticeTimeStep s)
          {
            seed = s;
          }

          void SetTemporalPeriod(const LatticeTimeStep p)
          {
            temporalPeriod = p;
          }

        private:
          static uint64_t Mix(uint64_t x);
          static double HashToUnit(uint64_t x);

          LatticeSpeed maxSpeed;
          Dimensionless intensity;
          LatticeTimeStep warmUpLength;
          LatticeTimeStep seed;
          LatticeTimeStep temporalPeriod;
      };
    }
  }
}

#endif // HEMELB_LB_IOLETS_INOUTLETTURBULENTVELOCITY_H

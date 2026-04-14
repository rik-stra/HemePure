// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/iolets/InOutLetTurbulentVelocity.h"

#include <cmath>
#include <cstdlib>

namespace hemelb
{
  namespace lb
  {
    namespace iolets
    {
      InOutLetTurbulentVelocity::InOutLetTurbulentVelocity() :
        maxSpeed(0.0), intensity(0.05), warmUpLength(0), seed(1), temporalPeriod(1)
      {
      }

      InOutLetTurbulentVelocity::~InOutLetTurbulentVelocity()
      {
      }

      InOutLet* InOutLetTurbulentVelocity::Clone() const
      {
        InOutLet* copy = new InOutLetTurbulentVelocity(*this);
        return copy;
      }

      uint64_t InOutLetTurbulentVelocity::Mix(uint64_t x)
      {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
      }

      double InOutLetTurbulentVelocity::HashToUnit(uint64_t x)
      {
        const uint64_t h = Mix(x);
        const uint64_t mantissa = h & ((1ULL << 53) - 1);
        return static_cast<double>(mantissa) / static_cast<double>(1ULL << 53);
      }

      LatticeVelocity InOutLetTurbulentVelocity::GetVelocity(const LatticePosition& x,
                                                             const LatticeTimeStep t) const
      {
        LatticePosition displ = x - position;
        LatticeDistance z = displ.Dot(normal);
        LatticeDistance rSq = displ.GetMagnitudeSquared() - z * z;
        Dimensionless rSqOverASq = rSq / (radius * radius);
        if (rSqOverASq > 1.0)
        {
          log::Logger::Log<log::Error, log::OnePerCore>(
            "An IOLET site with r = %lf lies outside the IOLET radius %lf.",
            std::sqrt(rSq), radius);
          std::exit(16);
        }

        LatticeSpeed currentMax = maxSpeed;
        if (warmUpLength > 0 && t < warmUpLength)
        {
          currentMax *= static_cast<double>(t) / static_cast<double>(warmUpLength);
        }

        const LatticeSpeed meanSpeed = currentMax * (1.0 - rSqOverASq);

        const LatticeTimeStep timeBucket = temporalPeriod > 0 ? (t / temporalPeriod) : t;
        const int64_t hx = static_cast<int64_t>(std::llround(x.x * 2.0));
        const int64_t hy = static_cast<int64_t>(std::llround(x.y * 2.0));
        const int64_t hz = static_cast<int64_t>(std::llround(x.z * 2.0));

        uint64_t key = static_cast<uint64_t>(seed);
        key ^= Mix(static_cast<uint64_t>(hx));
        key ^= Mix(static_cast<uint64_t>(hy) + 0x9e3779b97f4a7c15ULL);
        key ^= Mix(static_cast<uint64_t>(hz) + 0xbf58476d1ce4e5b9ULL);
        key ^= Mix(static_cast<uint64_t>(timeBucket) + 0x94d049bb133111ebULL);

        const double eta = 2.0 * HashToUnit(key) - 1.0;
        const LatticeSpeed turbulentSpeed = meanSpeed * (1.0 + intensity * eta);
        const LatticeSpeed speed = util::NumericalFunctions::max<LatticeSpeed>(0.0, turbulentSpeed);

        return normal * speed;
      }
    }
  }
}

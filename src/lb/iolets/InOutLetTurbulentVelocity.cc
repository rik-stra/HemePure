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
        maxSpeed(0.0), intensity(0.05), warmUpLength(0), seed(1), temporalPeriod(1),
        modeCount(0), minPeriod(20), maxPeriod(200), spectralExponent(5.0 / 3.0)
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
        const int64_t hx = static_cast<int64_t>(std::llround(x.x * 2.0));
        const int64_t hy = static_cast<int64_t>(std::llround(x.y * 2.0));
        const int64_t hz = static_cast<int64_t>(std::llround(x.z * 2.0));

        double eta = 0.0;
        if (modeCount > 0)
        {
          const LatticeTimeStep minP = minPeriod > 0 ? minPeriod : 1;
          const LatticeTimeStep maxP = maxPeriod >= minP ? maxPeriod : minP;

          const double pMin = static_cast<double>(minP);
          const double pMax = static_cast<double>(maxP);
          const double gamma = 0.5 * spectralExponent;

          double weighted = 0.0;
          double wsum = 0.0;
          const double twopi = 6.28318530717958647692;

          for (unsigned int m = 0; m < modeCount; ++m)
          {
            const double alpha = (modeCount > 1) ? static_cast<double>(m) / static_cast<double>(modeCount - 1) : 0.0;
            const double period = (pMin > 0.0) ? (pMin * std::pow(pMax / pMin, alpha)) : pMax;
            const double freq = period > 0.0 ? (1.0 / period) : 1.0;

            // Power-law target: S(f) ~ f^{-spectralExponent}; sinusoid amplitude A ~ f^{-spectralExponent/2}.
            const double w = std::pow(freq, -gamma);

            uint64_t key = static_cast<uint64_t>(seed);
            key ^= Mix(static_cast<uint64_t>(hx) + 0x9e3779b97f4a7c15ULL * (m + 1));
            key ^= Mix(static_cast<uint64_t>(hy) + 0xbf58476d1ce4e5b9ULL * (m + 3));
            key ^= Mix(static_cast<uint64_t>(hz) + 0x94d049bb133111ebULL * (m + 5));
            const double phase = twopi * HashToUnit(key);

            weighted += w * std::sin(twopi * freq * static_cast<double>(t) + phase);
            wsum += std::fabs(w);
          }

          eta = (wsum > 0.0) ? (weighted / wsum) : 0.0;
        }
        else
        {
          // Legacy mode: deterministic hash in space and coarse-grained time buckets.
          const LatticeTimeStep timeBucket = temporalPeriod > 0 ? (t / temporalPeriod) : t;
          uint64_t key = static_cast<uint64_t>(seed);
          key ^= Mix(static_cast<uint64_t>(hx));
          key ^= Mix(static_cast<uint64_t>(hy) + 0x9e3779b97f4a7c15ULL);
          key ^= Mix(static_cast<uint64_t>(hz) + 0xbf58476d1ce4e5b9ULL);
          key ^= Mix(static_cast<uint64_t>(timeBucket) + 0x94d049bb133111ebULL);
          eta = 2.0 * HashToUnit(key) - 1.0;
        }

        const LatticeSpeed turbulentSpeed = meanSpeed * (1.0 + intensity * eta);
        const LatticeSpeed speed = util::NumericalFunctions::max<LatticeSpeed>(0.0, turbulentSpeed);

        return normal * speed;
      }
    }
  }
}

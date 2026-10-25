// Example: correct the three tracks of one Ds -> K K pi candidate and the muon, then recompute m(K K pi).
// Build:  g++ -O2 -std=c++14 -I. example_standalone.cpp -o example && ./example ../data/rds_trkcalib_2018BParking_v2.json
#include "RDsTrackCalib.h"
#include <cmath>
#include <cstdio>

struct Track { double px, py, pz, m; int q; long long index; };  // index: position in the input track collection

static double invMass(const Track* t, const double* f, int n) {
  double E = 0, X = 0, Y = 0, Z = 0;
  for (int i = 0; i < n; ++i) {
    const double px = t[i].px * f[i], py = t[i].py * f[i], pz = t[i].pz * f[i];
    E += std::sqrt(px * px + py * py + pz * pz + t[i].m * t[i].m); X += px; Y += py; Z += pz;
  }
  return std::sqrt(std::max(E * E - X * X - Y * Y - Z * Z, 0.0));
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "../data/rds_trkcalib_2018BParking_v2.json";
  rdscalib::TrackCalib cal(path);
  const long long run = 320500, lumi = 117, event = 123456789;
  // one candidate (made-up momenta in GeV): K+, K-, pi+
  Track t[3] = {{1.20, 0.85, 0.40, 0.493677, +1, 12}, {2.10, 1.05, 0.95, 0.493677, -1, 37}, {3.40, 2.20, -0.60, 0.13957, +1, 5}};
  double f[3], fUp[3], one[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    const double pt = std::hypot(t[i].px, t[i].py), p = std::hypot(pt, t[i].pz), eta = std::asinh(t[i].pz / pt);
    const double z = cal.gauss(run, lumi, event, t[i].index);  // same track -> same z in every candidate
    f[i] = cal.factor(p, pt, eta, t[i].q, z);
    fUp[i] = cal.factor(p, pt, eta, t[i].q, z, "eig0_up");
    std::printf("track %d: pT %.3f GeV, eta %+.3f, |eta| bin %d, z %+.3f, factor %.6f\n", i, pt, eta, cal.etaBin(eta), z, f[i]);
  }
  std::printf("m(KKpi): raw %.4f GeV, corrected %.4f GeV, eig0_up %.4f GeV\n", invMass(t, one, 3), invMass(t, f, 3), invMass(t, fUp, 3));
  // the muon: same correction with its own pT, eta, charge
  const double muPt = 9.0, muEta = 0.8; const int muQ = -1; const long long muIndex = 1000000 + 2;  // offset: muons are another collection
  const double zMu = cal.gauss(run, lumi, event, muIndex);
  std::printf("muon pT %.3f -> %.4f GeV\n", muPt, cal.correctedPt(muPt, muEta, muQ, zMu));
  return 0;
}

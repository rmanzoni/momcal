// ROOT macro: add corrected track factors and the corrected m(K K pi) to an MC flat ntuple with RDataFrame.
//
//   root -l -b -q 'example_rdf.C("mc.root", "tree", "mc_trkcalib.root")'
//
// Edit the branch names in the Define strings to match your ntuple. Branch types (float/double, int/short)
// are converted by the string expressions, so no type matching is needed. The calibration object is const
// and thread safe: ROOT::EnableImplicitMT() can be used.
// Written for ROOT >= 6.24. Not run in the cloud test area (no ROOT there); the same calls are compiled and
// tested in example_standalone.cpp and test/crosscheck.cpp.
#include "RDsTrackCalib.h"
#include <ROOT/RDataFrame.hxx>
#include <cmath>
#include <string>

namespace rdsex {
const rdscalib::TrackCalib* cal = nullptr;

// factor p'/p of one track; random key = event + the track's own kinematics (same track -> same z)
double factor(double px, double py, double pz, int q, long long run, long long lumi, long long event, const char* variation) {
  const double pt = std::hypot(px, py), p = std::hypot(pt, pz), eta = std::asinh(pz / pt), phi = std::atan2(py, px);
  const double z = cal->gauss(run, lumi, event, rdscalib::trackIdFromKinematics(pt, eta, phi));
  return cal->factor(p, pt, eta, q, z, variation);
}

double mKKpi(double k1x, double k1y, double k1z, double f1, double k2x, double k2y, double k2z, double f2,
             double pix, double piy, double piz, double f3) {
  const double mk = 0.493677, mpi = 0.13957;
  auto e = [](double x, double y, double z, double m) { return std::sqrt(x * x + y * y + z * z + m * m); };
  const double E = e(k1x * f1, k1y * f1, k1z * f1, mk) + e(k2x * f2, k2y * f2, k2z * f2, mk) + e(pix * f3, piy * f3, piz * f3, mpi);
  const double X = k1x * f1 + k2x * f2 + pix * f3, Y = k1y * f1 + k2y * f2 + piy * f3, Z = k1z * f1 + k2z * f2 + piz * f3;
  return std::sqrt(std::max(E * E - X * X - Y * Y - Z * Z, 0.0));
}
}  // namespace rdsex

void example_rdf(const char* in, const char* tree, const char* out,
                 const char* json = "../data/rds_trkcalib_2018BParking_v1.json", bool hasEventId = false) {
  static rdscalib::TrackCalib calib(json);
  rdsex::cal = &calib;
  ROOT::RDataFrame df0(tree, in);
  ROOT::RDF::RNode df = hasEventId
      ? ROOT::RDF::RNode(df0.Define("ev_run", "(long long)run").Define("ev_lumi", "(long long)lumi").Define("ev_event", "(long long)event"))
      : ROOT::RDF::RNode(df0.Define("ev_run", "0LL").Define("ev_lumi", "0LL").Define("ev_event", "(long long)rdfentry_"));
  for (const std::string v : {"nominal", "eig0_up", "eig0_down"}) {
    const std::string sfx = v == "nominal" ? "" : "_" + v;
    for (const std::string t : {"k1", "k2", "pi"})
      df = df.Define(t + "_f" + sfx, "rdsex::factor(" + t + "_px, " + t + "_py, " + t + "_pz, " + t +
                                         "_charge, ev_run, ev_lumi, ev_event, \"" + v + "\")");
    df = df.Define("kkpi_m" + sfx, "rdsex::mKKpi(k1_px, k1_py, k1_pz, k1_f" + sfx + ", k2_px, k2_py, k2_pz, k2_f" + sfx +
                                       ", pi_px, pi_py, pi_pz, pi_f" + sfx + ")");
  }
  df.Snapshot("trkcalib", out, "(k1|k2|pi)_f.*|kkpi_m.*");
}

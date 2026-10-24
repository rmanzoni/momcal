// Reads tracks.csv (run,lumi,event,track_id,p,pt,eta,q,pt_raw,eta_raw,phi_raw), writes cpp_out.csv with
// track_id_kin, z, and the factor for every variation. Build: g++ -O2 -std=c++14 -I../cpp crosscheck.cpp -o crosscheck
#include "RDsTrackCalib.h"
#include <cstdio>
#include <iostream>
int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: crosscheck <json> <tracks.csv> <out.csv>\n"; return 1; }
  rdscalib::TrackCalib cal(argv[1]);
  std::ifstream in(argv[2]);
  FILE* out = std::fopen(argv[3], "w");
  const auto vars = cal.variations();
  std::fprintf(out, "track_id_kin,z");
  for (const auto& v : vars) std::fprintf(out, ",%s", v.c_str());
  std::fprintf(out, "\n");
  std::string line;
  std::getline(in, line);  // header
  long n = 0;
  while (std::getline(in, line)) {
    long long run, lumi, event, tid; double p, pt, eta, ptr, etar, phir; int q;
    if (std::sscanf(line.c_str(), "%lld,%lld,%lld,%lld,%lf,%lf,%lf,%d,%lf,%lf,%lf", &run, &lumi, &event, &tid, &p, &pt, &eta, &q, &ptr, &etar, &phir) != 11) continue;
    const double z = cal.gauss(run, lumi, event, tid);
    std::fprintf(out, "%lld,%.17g", static_cast<long long>(rdscalib::trackIdFromKinematics(ptr, etar, phir)), z);
    for (const auto& v : vars) std::fprintf(out, ",%.17g", cal.factor(p, pt, eta, q, z, v));
    std::fprintf(out, "\n");
    ++n;
  }
  std::fclose(out);
  std::cout << cal.name() << ": " << n << " tracks, " << vars.size() << " variations\n";
  return 0;
}

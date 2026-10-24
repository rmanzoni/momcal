// RDsTrackCalib.h -- apply the R(Ds) track momentum calibration (schema rds_trkcalib/1) to MC tracks.
//
//   p' = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z)
//
// Only the momentum magnitude changes; the direction is kept. Apply to MC only, to every charged track
// (kaons, pion, muon), each with its own pT, charge and |eta|.
//
// Header only, C++14, no dependencies (a small JSON reader is included). Works in CMSSW, in ROOT macros
// (also with RDataFrame multithreading: all methods are const) and in plain executables.
//
//   #include "RDsTrackCalib.h"
//   rdscalib::TrackCalib cal("rds_trkcalib_2018BParking_v1.json");
//   double z = cal.gauss(run, lumi, event, trackId);        // fixed per track
//   double f = cal.factor(p, pt, eta, q, z);                // p'/p
//   double fUp = cal.factor(p, pt, eta, q, z, "eig0_up");   // statistical variations eig0 ... eig10
//
// The Python module rds_trkcalib.py gives identical numbers (see test/).
#ifndef RDS_TRACK_CALIB_H
#define RDS_TRACK_CALIB_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rdscalib {

// ------------------------------------------------------------------ minimal JSON reader
namespace json {
struct Value {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;
  const Value& operator[](const std::string& k) const {
    auto it = obj.find(k);
    if (type != Object || it == obj.end()) throw std::runtime_error("rdscalib json: missing key '" + k + "'");
    return it->second;
  }
  const Value& operator[](size_t i) const {
    if (type != Array || i >= arr.size()) throw std::runtime_error("rdscalib json: bad array index");
    return arr[i];
  }
  bool has(const std::string& k) const { return type == Object && obj.count(k); }
  std::vector<double> numbers() const {
    std::vector<double> v;
    for (const auto& x : arr) v.push_back(x.num);
    return v;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  Value parse() {
    Value v = value();
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;
  [[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("rdscalib json: " + m + " at offset " + std::to_string(i_));
  }
  void ws() { while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_; }
  char peek() { ws(); if (i_ >= s_.size()) fail("unexpected end"); return s_[i_]; }
  void expect(char c) { if (peek() != c) fail(std::string("expected '") + c + "'"); ++i_; }
  bool lit(const char* w) {
    size_t n = std::char_traits<char>::length(w);
    if (s_.compare(i_, n, w) == 0) { i_ += n; return true; }
    return false;
  }
  Value value() {
    char c = peek();
    Value v;
    if (c == '{') {
      v.type = Value::Object; ++i_;
      if (peek() == '}') { ++i_; return v; }
      while (true) {
        std::string k = string();
        expect(':');
        v.obj[k] = value();
        char d = peek(); ++i_;
        if (d == '}') break;
        if (d != ',') fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.type = Value::Array; ++i_;
      if (peek() == ']') { ++i_; return v; }
      while (true) {
        v.arr.push_back(value());
        char d = peek(); ++i_;
        if (d == ']') break;
        if (d != ',') fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.type = Value::String; v.str = string();
    } else if (lit("true")) { v.type = Value::Bool; v.b = true;
    } else if (lit("false")) { v.type = Value::Bool;
    } else if (lit("null")) { v.type = Value::Null;
    } else if (lit("NaN")) { v.type = Value::Number; v.num = std::nan("");
    } else if (lit("Infinity")) { v.type = Value::Number; v.num = HUGE_VAL;
    } else if (lit("-Infinity")) { v.type = Value::Number; v.num = -HUGE_VAL;
    } else {
      size_t j = i_;
      while (j < s_.size() && std::string("+-0123456789.eE").find(s_[j]) != std::string::npos) ++j;
      if (j == i_) fail("unexpected character");
      v.type = Value::Number;
      std::istringstream is(s_.substr(i_, j - i_));
      is.imbue(std::locale::classic());
      is >> v.num;
      if (is.fail()) fail("bad number");
      i_ = j;
    }
    return v;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("bad escape");
        char e = s_[i_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {  // keep ASCII, replace anything else by '?': strings are only descriptive here
            unsigned cp = std::stoul(s_.substr(i_, 4), nullptr, 16); i_ += 4;
            out += cp < 128 ? static_cast<char>(cp) : '?';
            break;
          }
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    expect('"');
    return out;
  }
};

inline Value parseFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("rdscalib: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  return Parser(s).parse();
}
}  // namespace json

// ------------------------------------------------------------------ deterministic random numbers
inline uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline double unit53(uint64_t h) { return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

// Unit Gaussian fixed by (seed, run, lumi, event, trackId). Signed inputs wrap as two's complement.
inline double gaussFromKey(uint64_t seed, int64_t run, int64_t lumi, int64_t event, int64_t trackId) {
  uint64_t h = splitmix64(seed);
  h = splitmix64(h ^ static_cast<uint64_t>(run));
  h = splitmix64(h ^ static_cast<uint64_t>(lumi));
  h = splitmix64(h ^ static_cast<uint64_t>(event));
  h = splitmix64(h ^ static_cast<uint64_t>(trackId));
  const uint64_t h2 = splitmix64(h);
  const double u1 = unit53(h), u2 = unit53(h2);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.141592653589793 * u2);
}

// Integer track identifier from (pT [GeV], eta, phi) for ntuples without a track index:
// pT rounded to 1 MeV, eta and phi to 1e-4, packed into 58 bits (24 + 17 + 17).
inline int64_t trackIdFromKinematics(double pt, double eta, double phi) {
  const int64_t ipt = static_cast<int64_t>(std::floor(pt * 1000.0 + 0.5)) & ((1LL << 24) - 1);
  const int64_t ie = static_cast<int64_t>(std::floor((eta + 6.5) * 1e4 + 0.5)) & ((1LL << 17) - 1);
  const int64_t ip = static_cast<int64_t>(std::floor((phi + 3.2) * 1e4 + 0.5)) & ((1LL << 17) - 1);
  return (ipt << 34) | (ie << 17) | ip;
}

// ------------------------------------------------------------------ the correction
struct Params {
  std::vector<double> s0, s1, k, v0, v1;  // one value per |eta| bin
};

class TrackCalib {
 public:
  explicit TrackCalib(const std::string& path) {
    const json::Value c = json::parseFile(path);
    if (c["schema"].str != "rds_trkcalib/1") throw std::runtime_error("rdscalib: unknown schema in " + path);
    name_ = c["name"].str + " " + c["version"].str;
    edges_ = c["eta_bins"]["edges"].numbers();
    nb_ = static_cast<int>(edges_.size()) - 1;
    seed_ = static_cast<uint64_t>(c["random"]["seed"].num);
    const json::Value& pb = c["per_bin"];
    Params nom{pb["s0"].numbers(), pb["s1"].numbers(), pb["k"].numbers(), pb["v0"].numbers(), pb["v1"].numbers()};
    for (const auto* v : {&nom.s0, &nom.s1, &nom.k, &nom.v0, &nom.v1})
      if (static_cast<int>(v->size()) != nb_) throw std::runtime_error("rdscalib: per_bin size mismatch");
    sets_["nominal"] = nom;
    Params so = nom; std::fill(so.v0.begin(), so.v0.end(), 0.); std::fill(so.v1.begin(), so.v1.end(), 0.);
    sets_["scale_only"] = so;
    Params sm = nom; for (auto* v : {&sm.s0, &sm.s1, &sm.k}) std::fill(v->begin(), v->end(), 0.);
    sets_["smear_only"] = sm;
    const json::Value& var = c["variations"];
    for (size_t i = 0; i < var["names"].arr.size(); ++i) {
      const std::string n = var["names"][i].str;
      const json::Value& s = var["shift_per_bin"][i];
      for (int sign : {+1, -1}) {
        Params p = nom;
        auto add = [&](std::vector<double>& t, const json::Value& d) {
          for (int b = 0; b < nb_; ++b) t[b] += sign * d[b].num;
        };
        add(p.s0, s["s0"]); add(p.s1, s["s1"]); add(p.k, s["k"]); add(p.v0, s["v0"]); add(p.v1, s["v1"]);
        sets_[n + (sign > 0 ? "_up" : "_down")] = p;
      }
    }
  }

  const std::string& name() const { return name_; }
  int nBins() const { return nb_; }
  const std::vector<double>& etaEdges() const { return edges_; }
  std::vector<std::string> variations() const {
    std::vector<std::string> v;
    for (const auto& kv : sets_) v.push_back(kv.first);
    return v;
  }
  const Params& params(const std::string& variation = "nominal") const {
    auto it = sets_.find(variation);
    if (it == sets_.end()) throw std::runtime_error("rdscalib: unknown variation '" + variation + "'");
    return it->second;
  }

  // |eta| bin; beyond the last edge -> last bin
  int etaBin(double eta) const {
    const double a = std::fabs(eta);
    int b = static_cast<int>(std::upper_bound(edges_.begin(), edges_.end(), a) - edges_.begin()) - 1;
    return std::min(std::max(b, 0), nb_ - 1);
  }

  // scale factor 1 + s0 + s1/p + k q pT
  double scale(double p, double pt, double eta, int q, const std::string& variation = "nominal") const {
    const Params& P = params(variation);
    const int b = etaBin(eta);
    return 1.0 + P.s0[b] + P.s1[b] / p + P.k[b] * q * pt;
  }
  // extra relative resolution sqrt(v0 1e-6 + v1 1e-8 pT^2)
  double resolution(double pt, double eta, const std::string& variation = "nominal") const {
    const Params& P = params(variation);
    const int b = etaBin(eta);
    return std::sqrt(std::max(P.v0[b] * 1e-6 + P.v1[b] * 1e-8 * pt * pt, 0.0));
  }
  // full factor p'/p
  double factor(double p, double pt, double eta, int q, double z, const std::string& variation = "nominal") const {
    return scale(p, pt, eta, q, variation) * (1.0 + resolution(pt, eta, variation) * z);
  }
  // the per-track Gaussian z with this file's seed
  double gauss(int64_t run, int64_t lumi, int64_t event, int64_t trackId) const {
    return gaussFromKey(seed_, run, lumi, event, trackId);
  }
  // convenience: corrected pT from (pT, eta); direction unchanged
  double correctedPt(double pt, double eta, int q, double z, const std::string& variation = "nominal") const {
    return pt * factor(pt * std::cosh(eta), pt, eta, q, z, variation);
  }

 private:
  std::string name_;
  std::vector<double> edges_;
  int nb_ = 0;
  uint64_t seed_ = 0;
  std::map<std::string, Params> sets_;
};

}  // namespace rdscalib
#endif

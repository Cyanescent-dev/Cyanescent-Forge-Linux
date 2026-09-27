//
//  Json.hpp
//  Cyanescent Forge — Linux worker
//
//  A minimal ordered JSON value for benchmark.json, --capabilities and
//  --snapshot-hashes. Keys are written sorted (like the macOS worker's
//  .sortedKeys) and doubles with 17 significant digits.
//
#ifndef FORGE_JSON_HPP
#define FORGE_JSON_HPP

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace forge {

class Json {
public:
    enum class Kind { Null, Bool, Int, Double, String, Array, Object };

    Json() : kind_(Kind::Null) {}
    Json(std::nullptr_t) : kind_(Kind::Null) {}
    Json(bool v) : kind_(Kind::Bool), b_(v) {}
    Json(int v) : kind_(Kind::Int), i_(v) {}
    Json(long v) : kind_(Kind::Int), i_(v) {}
    Json(long long v) : kind_(Kind::Int), i_(v) {}
    Json(unsigned v) : kind_(Kind::Int), i_(v) {}
    Json(unsigned long v) : kind_(Kind::Int), i_(static_cast<long long>(v)) {}
    Json(unsigned long long v) : kind_(Kind::Int), i_(static_cast<long long>(v)) {}
    Json(double v) : kind_(Kind::Double), d_(v) {}
    Json(const char *v) : kind_(Kind::String), s_(v) {}
    Json(std::string v) : kind_(Kind::String), s_(std::move(v)) {}

    static Json array() { Json j; j.kind_ = Kind::Array; return j; }
    static Json object() { Json j; j.kind_ = Kind::Object; return j; }

    Json &operator[](const std::string &key) {
        kind_ = Kind::Object;
        return o_[key];
    }
    void push(Json v) {
        kind_ = Kind::Array;
        a_.push_back(std::move(v));
    }

    std::string dump(int indent = 2) const {
        std::string out;
        write(out, indent, 0);
        return out;
    }

private:
    static void escape(std::string &out, const std::string &s) {
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
            }
        }
        out.push_back('"');
    }

    void write(std::string &out, int indent, int depth) const {
        const std::string pad(static_cast<size_t>(indent * (depth + 1)), ' ');
        const std::string closePad(static_cast<size_t>(indent * depth), ' ');
        switch (kind_) {
        case Kind::Null: out += "null"; break;
        case Kind::Bool: out += b_ ? "true" : "false"; break;
        case Kind::Int: out += std::to_string(i_); break;
        case Kind::Double: {
            if (!std::isfinite(d_)) { out += "null"; break; }
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", d_);
            out += buf;
            break;
        }
        case Kind::String: escape(out, s_); break;
        case Kind::Array:
            if (a_.empty()) { out += "[]"; break; }
            out += "[\n";
            for (size_t i = 0; i < a_.size(); ++i) {
                out += pad;
                a_[i].write(out, indent, depth + 1);
                out += i + 1 < a_.size() ? ",\n" : "\n";
            }
            out += closePad + "]";
            break;
        case Kind::Object: {
            if (o_.empty()) { out += "{}"; break; }
            out += "{\n";
            size_t i = 0;
            for (const auto &kv : o_) {
                out += pad;
                escape(out, kv.first);
                out += " : ";
                kv.second.write(out, indent, depth + 1);
                out += ++i < o_.size() ? ",\n" : "\n";
            }
            out += closePad + "}";
            break;
        }
        }
    }

    Kind kind_;
    bool b_ = false;
    long long i_ = 0;
    double d_ = 0;
    std::string s_;
    std::vector<Json> a_;
    std::map<std::string, Json> o_;
};

} // namespace forge

#endif // FORGE_JSON_HPP

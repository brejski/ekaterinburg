#ifndef EKATERINBURG_EFFECTS_H
#define EKATERINBURG_EFFECTS_H

#include <imgui.h>
#include <map>
#include <string>

namespace effects {

void FX_circles(ImDrawList *d,
                ImVec2 a,
                ImVec2 b,
                ImVec2 sz,
                ImVec4 mouse,
                float t);

void FX_blobs(ImDrawList *d, ImVec2 a, ImVec2 b, ImVec2 s, ImVec4 m, float t);

const std::map<const char*,
               std::function<void(ImDrawList *,
                                  ImVec2,
                                  ImVec2,
                                  ImVec2,
                                  ImVec4,
                                  float)>> effects_map {
        {"circles", FX_circles},
        {"blobs", FX_blobs}
};
}  // namespace effects

#endif //EKATERINBURG_EFFECTS_H

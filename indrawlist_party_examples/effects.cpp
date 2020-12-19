#include "effects.h"
#include <cmath>

namespace effects {
void FX_circles(ImDrawList *d, ImVec2 a, ImVec2 b, ImVec2 sz, ImVec4 mouse, float t) {
    for (int n = 0; n < (1.0f + sinf(t * 5.7f)) * 40.0f; n++)
        d->AddCircle(ImVec2(a.x + sz.x * 0.5f, a.y + sz.y * 0.5f),
                     sz.y * (0.01f + n * 0.03f),
                     IM_COL32(255, 140 - n * 4, n * 3, 255));
}

void FX_blobs(ImDrawList *d, ImVec2 a, ImVec2 b, ImVec2 s, ImVec4 m, float t) {
    int N = 25;
    float sp = s.y / N, y, st = sin(t) * 0.5 + 0.5,
            r[3] = {1500, 1087 + 200 * st, 1650},
            ctr[3][2] = {{150, 140}, {s.x * m.x, s.y * m.y},
                         {40 + 200 * st, 73 + 20 * sin(st * 5)}};
    for (int i = 0; i < N; i++) {
        y = a.y + sp * (i + .5);
        for (int x = a.x; x <= b.x; x += 2) {
            float D = 0, o = 255;
            for (int k = 0; k < 3; k++)
                D += r[k] / (pow(x - a.x - ctr[k][0], 2)
                        + pow(y - a.y - ctr[k][1], 2));
            if (D < 1) continue;
            if (D > 2.5) D = 2.5;
            if (D < 1.15) o /= 2;
            d->AddLine(ImVec2(x, y),
                       ImVec2(x + 2, y),
                       IM_COL32(239, 129, 19, o),
                       D + sin(2.3 * t + 0.3 * i));
        }
    }
}
}  // namespace effects
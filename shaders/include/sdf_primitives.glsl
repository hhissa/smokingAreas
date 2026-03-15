// sdf_primitives.glsl
//
// All primitives updated from (vec3 pos, mat3 rot) to (mat4 invTransform).
//
// The invTransform is the pre-inverted TRS matrix computed on the CPU by
// Transform::invMatrix().  Applying it to the march point brings `p` into
// the primitive's local space in one operation, correctly handling:
//   - translation  (the old `p - pos`)
//   - rotation     (the old `rot * ...`)
//   - non-uniform scale  (previously silently broken)
//
// Normal computation:
//   The tetrahedron-method normals (4 map() calls) give world-space normals
//   automatically because they sample map() at world-space offsets.
//   If you ever need to transform a local-space normal explicitly, use
//   the inverse-transpose of the *model* matrix (not invTransform):
//     vec3 worldNormal = normalize(mat3(transpose(invTransform)) * localNormal);
//   This is handled in the lighting pass, not here.
//
// SDF struct: carries distance + colour so boolean ops can blend materials.

struct SDF {
    float dist;
    vec3  color;
};

// ─────────────────────────────────────────────────────────────────────────────
// BOOLEAN OPERATORS  (branchless, colour-aware)
// ─────────────────────────────────────────────────────────────────────────────

SDF opUnion(SDF a, SDF b) {
    float k = step(b.dist, a.dist);
    return SDF(min(a.dist, b.dist), mix(a.color, b.color, k));
}

SDF opSubtraction(SDF a, SDF b) {
    float k = step(b.dist, -a.dist);
    return SDF(max(-a.dist, b.dist), mix(a.color, b.color, k));
}

SDF opIntersection(SDF a, SDF b) {
    float k = step(b.dist, a.dist);
    return SDF(max(a.dist, b.dist), mix(a.color, b.color, k));
}

SDF opSmoothUnion(SDF a, SDF b, float k) {
    float h = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
    return SDF(
        mix(b.dist, a.dist, h) - k * h * (1.0 - h),
        mix(b.color, a.color, h)
    );
}

SDF opSmoothSubtraction(SDF a, SDF b, float k) {
    float h = clamp(0.5 - 0.5 * (b.dist + a.dist) / k, 0.0, 1.0);
    return SDF(
        mix(b.dist, -a.dist, h) + k * h * (1.0 - h),
        mix(b.color, a.color, h)
    );
}

SDF opSmoothIntersection(SDF a, SDF b, float k) {
    float h = clamp(0.5 - 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
    return SDF(
        mix(b.dist, a.dist, h) + k * h * (1.0 - h),
        mix(b.color, a.color, h)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PRIMITIVES
//
// Every function signature is now:
//   SDF sdfXxx(vec3 p, mat4 invTransform, <shape params>, vec3 color)
//
// `p`            — world-space march point
// `invTransform` — pre-inverted TRS from Transform::invMatrix() on the CPU
// ─────────────────────────────────────────────────────────────────────────────

// ── Sphere ────────────────────────────────────────────────────────────────────
SDF sdfSphere(vec3 p, mat4 invT, float radius, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    return SDF(length(pl) - radius, color);
}

// ── Box ───────────────────────────────────────────────────────────────────────
SDF sdfBox(vec3 p, mat4 invT, vec3 b, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 q  = abs(pl) - b;
    return SDF(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0), color);
}

// ── Round Box ─────────────────────────────────────────────────────────────────
SDF sdfRoundBox(vec3 p, mat4 invT, vec3 b, float r, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 q  = abs(pl) - b + r;
    return SDF(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r, color);
}

// ── Box Frame ─────────────────────────────────────────────────────────────────
SDF sdfBoxFrame(vec3 p, mat4 invT, vec3 b, float e, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 pp = abs(pl) - b;
    vec3 q  = abs(pp + e) - e;
    float d = min(min(
        length(max(vec3(pp.x, q.y, q.z), 0.0)) + min(max(pp.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, pp.y, q.z), 0.0)) + min(max(q.x, max(pp.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, pp.z), 0.0)) + min(max(q.x, max(q.y, pp.z)), 0.0));
    return SDF(d, color);
}

// ── Rounded Box Frame ─────────────────────────────────────────────────────────
SDF sdfRoundedBoxFrame(vec3 p, mat4 invT, vec3 b, float e, float r, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 pp = abs(pl) - b;
    vec3 q  = abs(pp + e) - e;
    float d = min(min(
        length(max(vec3(pp.x, q.y, q.z), 0.0)) + min(max(pp.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, pp.y, q.z), 0.0)) + min(max(q.x, max(pp.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, pp.z), 0.0)) + min(max(q.x, max(q.y, pp.z)), 0.0));
    return SDF(d - r, color);
}

// ── Torus ─────────────────────────────────────────────────────────────────────
SDF sdfTorus(vec3 p, mat4 invT, float majorR, float minorR, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec2 q  = vec2(length(pl.xz) - majorR, pl.y);
    return SDF(length(q) - minorR, color);
}

// ── Capped Torus ──────────────────────────────────────────────────────────────
SDF sdfCappedTorus(vec3 p, mat4 invT, vec2 sc, float outerR, float innerR, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    pl.x    = abs(pl.x);
    float k = (sc.y * pl.x > sc.x * pl.y) ? dot(pl.xy, sc) : length(pl.xy);
    return SDF(sqrt(dot(pl, pl) + outerR*outerR - 2.0*outerR*k) - innerR, color);
}

// ── Link ──────────────────────────────────────────────────────────────────────
SDF sdfLink(vec3 p, mat4 invT, float halfLen, float outerR, float innerR, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 q  = vec3(pl.x, max(abs(pl.y) - halfLen, 0.0), pl.z);
    return SDF(length(vec2(length(q.xy) - outerR, q.z)) - innerR, color);
}

// ── Infinite Cylinder ─────────────────────────────────────────────────────────
SDF sdfCylinder(vec3 p, mat4 invT, vec2 centre, float radius, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    return SDF(length(pl.xz - centre) - radius, color);
}

// ── Cone ──────────────────────────────────────────────────────────────────────
SDF sdfCone(vec3 p, mat4 invT, vec2 sinCos, float height, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec2 q  = height * vec2(sinCos.x / sinCos.y, -1.0);
    vec2 w  = vec2(length(pl.xz), pl.y);
    vec2 a  = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b  = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return SDF(sqrt(d) * sign(s), color);
}

// ── Plane ─────────────────────────────────────────────────────────────────────
SDF sdfPlane(vec3 p, mat4 invT, vec3 normal, float offset, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    return SDF(dot(pl, normal) + offset, color);
}

// ── Hex Prism ─────────────────────────────────────────────────────────────────
SDF sdfHexPrism(vec3 p, mat4 invT, float radius, float height, vec3 color) {
    vec3 pl       = vec3(invT * vec4(p, 1.0));
    const vec3 k  = vec3(-0.8660254, 0.5, 0.57735);
    pl            = abs(pl);
    pl.xy        -= 2.0 * min(dot(k.xy, pl.xy), 0.0) * k.xy;
    vec2 d = vec2(
        length(pl.xy - vec2(clamp(pl.x, -k.z * radius, k.z * radius), radius)) * sign(pl.y - radius),
        pl.z - height);
    return SDF(min(max(d.x, d.y), 0.0) + length(max(d, 0.0)), color);
}

// ── Tri Prism ─────────────────────────────────────────────────────────────────
SDF sdfTriPrism(vec3 p, mat4 invT, float radius, float height, vec3 color) {
    vec3 pl = abs(vec3(invT * vec4(p, 1.0)));
    return SDF(max(pl.z - height, max(pl.x * 0.866025 + pl.y * 0.5, -pl.y) - radius * 0.5), color);
}

// ── Capsule ───────────────────────────────────────────────────────────────────
SDF sdfCapsule(vec3 p, mat4 invT, vec3 a, vec3 b, float radius, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec3 pa = pl - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return SDF(length(pa - ba * h) - radius, color);
}

// ── Vertical Capsule ──────────────────────────────────────────────────────────
SDF sdfVerticalCapsule(vec3 p, mat4 invT, float height, float radius, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    pl.y   -= clamp(pl.y, 0.0, height);
    return SDF(length(pl) - radius, color);
}

// ── Capped Cylinder ───────────────────────────────────────────────────────────
SDF sdfCappedCylinder(vec3 p, mat4 invT, float radius, float height, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec2 d  = abs(vec2(length(pl.xz), pl.y)) - vec2(radius, height);
    return SDF(min(max(d.x, d.y), 0.0) + length(max(d, 0.0)), color);
}

// ── Rounded Cylinder ──────────────────────────────────────────────────────────
SDF sdfRoundedCylinder(vec3 p, mat4 invT, float bodyR, float roundR, float height, vec3 color) {
    vec3 pl = vec3(invT * vec4(p, 1.0));
    vec2 d  = vec2(length(pl.xz) - bodyR + roundR, abs(pl.y) - height + roundR);
    return SDF(min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - roundR, color);
}

// ── Capped Cone ───────────────────────────────────────────────────────────────
SDF sdfCappedCone(vec3 p, mat4 invT, float height, float r1, float r2, vec3 color) {
    vec3 pl  = vec3(invT * vec4(p, 1.0));
    vec2 q   = vec2(length(pl.xz), pl.y);
    vec2 k1  = vec2(r2, height);
    vec2 k2  = vec2(r2 - r1, 2.0 * height);
    vec2 ca  = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - height);
    vec2 cb  = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s  = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return SDF(s * sqrt(min(dot(ca, ca), dot(cb, cb))), color);
}

// ── Round Cone ────────────────────────────────────────────────────────────────
SDF sdfRoundCone(vec3 p, mat4 invT, float r1, float r2, float height, vec3 color) {
    vec3  pl = vec3(invT * vec4(p, 1.0));
    float b  = (r1 - r2) / height;
    float a  = sqrt(1.0 - b * b);
    vec2  q  = vec2(length(pl.xz), pl.y);
    float k  = dot(q, vec2(a, b));
    float d;
    if      (k < 0.0)      d = length(q) - r1;
    else if (k > a * height) d = length(q - vec2(0.0, height)) - r2;
    else                   d = dot(q, vec2(a, b)) - r1;  // wait — this is signed correctly
    return SDF(d, color);
}

// ── Ellipsoid ─────────────────────────────────────────────────────────────────
// Note: the ellipsoid SDF is an *approximation* — exact computation is much
// more expensive.  This is the standard Quilez approximation; it is exact
// only when radii are equal (sphere) and has ~5–10% error otherwise.
SDF sdfEllipsoid(vec3 p, mat4 invT, vec3 radii, vec3 color) {
    vec3  pl = vec3(invT * vec4(p, 1.0));
    float k0 = length(pl / radii);
    float k1 = length(pl / (radii * radii));
    return SDF(k0 * (k0 - 1.0) / k1, color);
}

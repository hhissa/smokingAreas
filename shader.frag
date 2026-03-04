
#version 450

#define EPSILON 0.0001
#define MAX_STEPS 500
#define MAX_DISTANCE 1000.0
#define MIN_DISTANCE 0.0001
precision highp float;
precision highp int;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  vec2 resolution;
  float time;
  float starttime;
  int state;
} pc;

struct Light {
  vec3 position;
  vec3 direction;
  vec4 color;
  float brightness;
  float penumbraFactor;
} light;

struct RayInfo {
  vec3 origin;
  vec3 dir;
};

// SDF struct with color
struct SDF {
  float dist; // distance to surface
  vec3 color; // associated color
};

float hash(float n) {
  return fract(sin(n) * 43758.5453);
}

float hash(vec3 p) {
  return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

float noise(vec3 x) {
  vec3 p = floor(x);
  vec3 f = fract(x);

  f = f * f * (3.0 - 2.0 * f);

  float n = p.x + p.y * 57.0 + 113.0 * p.z;

  return mix(
    mix(
      mix(hash(n + 0.0), hash(n + 1.0), f.x),
      mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y
    ),
    mix(
      mix(hash(n + 113.0), hash(n + 114.0), f.x),
      mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y
    ),
    f.z
  );
}

mat3 rotatey(float theta) {
  return mat3(vec3(cos(theta), 0.0, sin(theta)),
    vec3(0.0, 1.0, 0.0),
    vec3(-sin(theta), 0.0, cos(theta)));
}

mat3 rotatex(float theta) {
  return mat3(vec3(1.0, 0.0, 0.0),
    vec3(0.0, cos(theta), -sin(theta)),
    vec3(0.0, sin(theta), cos(theta)));
  ;
}

mat3 rotatez(float theta) {
  return mat3(
    vec3(cos(theta), -sin(theta), 0.0),
    vec3(sin(theta), cos(theta), 0.0),
    vec3(0.0, 0.0, 1.0)
  );
}

///////////////////////////////////////////////////////////////////////////////////////
// INIT FUNCTIONS //

void initLight() {
  light.position = vec3(0.0, 0.0, 0.0);
  light.direction = vec3(0.0, 0.3, -1.0);
}

///////////////////////////////////////////////////////////////////////////////////////
// BOOLEAN OPERATORS (branchless, color-aware) //
// Union
SDF opUnion(SDF a, SDF b) {
  float k = step(b.dist, a.dist);
  SDF outSDF;
  outSDF.dist = min(a.dist, b.dist);
  outSDF.color = mix(a.color, b.color, k);
  return outSDF;
}

// Subtraction
SDF opSubtraction(SDF a, SDF b) {
  float d = max(-a.dist, b.dist);
  float k = step(b.dist, -a.dist);
  SDF outSDF;
  outSDF.dist = d;
  outSDF.color = mix(a.color, b.color, k);
  return outSDF;
}

// Intersection
SDF opIntersection(SDF a, SDF b) {
  float d = max(a.dist, b.dist);
  float k = step(b.dist, a.dist);
  SDF outSDF;
  outSDF.dist = d;
  outSDF.color = mix(a.color, b.color, k);
  return outSDF;
}

// Smooth Union
SDF opSmoothUnion(SDF a, SDF b, float k) {
  float h = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
  SDF outSDF;
  outSDF.dist = mix(b.dist, a.dist, h) - k * h * (1.0 - h);
  outSDF.color = mix(b.color, a.color, h);
  return outSDF;
}

// Smooth Subtraction
SDF opSmoothSubtraction(SDF a, SDF b, float k) {
  float h = clamp(0.5 - 0.5 * (b.dist + a.dist) / k, 0.0, 1.0);
  SDF outSDF;
  outSDF.dist = mix(b.dist, -a.dist, h) + k * h * (1.0 - h);
  outSDF.color = mix(b.color, a.color, h);
  return outSDF;
}

// Smooth Intersection
SDF opSmoothIntersection(SDF a, SDF b, float k) {
  float h = clamp(0.5 - 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
  SDF outSDF;
  outSDF.dist = mix(b.dist, a.dist, h) + k * h * (1.0 - h);
  outSDF.color = mix(b.color, a.color, h);
  return outSDF;
}

///////////////////////////////////////////////////////////////////////////////////////
// PRIMITIVES //

SDF sdfSphere(vec3 p, vec3 pos, mat3 rot, float s, vec3 color) {
  vec3 pl = rot * (p - pos);
  SDF o;
  o.dist = length(pl) - s;
  o.color = color;
  return o;
}

SDF sdfBox(vec3 p, vec3 pos, mat3 rot, vec3 b, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 q = abs(pl) - b;
  SDF o;
  o.dist = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  o.color = color;
  return o;
}

SDF sdfRoundBox(vec3 p, vec3 pos, mat3 rot, vec3 b, float r, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 q = abs(pl) - b + r;
  SDF o;
  o.dist = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
  o.color = color;
  return o;
}

SDF sdfBoxFrame(vec3 p, vec3 pos, mat3 rot, vec3 b, float e, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 pp = abs(pl) - b;
  vec3 q = abs(pp + e) - e;
  SDF o;
  o.dist = min(min(
        length(max(vec3(pp.x, q.y, q.z), 0.0)) + min(max(pp.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, pp.y, q.z), 0.0)) + min(max(q.x, max(pp.y, q.z)), 0.0)),
      length(max(vec3(q.x, q.y, pp.z), 0.0)) + min(max(q.x, max(q.y, pp.z)), 0.0));
  o.color = color;
  return o;
}

SDF sdfRoundedBoxFrame(vec3 p, vec3 pos, mat3 rot, vec3 b, float e, float r, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 pp = abs(pl) - b;
  vec3 q = abs(pp + e) - e;
  float d = min(min(
        length(max(vec3(pp.x, q.y, q.z), 0.0)) + min(max(pp.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, pp.y, q.z), 0.0)) + min(max(q.x, max(pp.y, q.z)), 0.0)),
      length(max(vec3(q.x, q.y, pp.z), 0.0)) + min(max(q.x, max(q.y, pp.z)), 0.0)
    );
  SDF o;
  o.dist = d - r;
  o.color = color;
  return o;
}

SDF sdfTorus(vec3 p, vec3 pos, mat3 rot, vec2 t, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec2 q = vec2(length(pl.xz) - t.x, pl.y);
  SDF o;
  o.dist = length(q) - t.y;
  o.color = color;
  return o;
}

SDF sdfCappedTorus(vec3 p, vec3 pos, mat3 rot, vec2 sc, float ra, float rb, vec3 color) {
  vec3 pl = rot * (p - pos);
  pl.x = abs(pl.x);
  float k = (sc.y * pl.x > sc.x * pl.y) ? dot(pl.xy, sc) : length(pl.xy);
  SDF o;
  o.dist = sqrt(dot(pl, pl) + ra * ra - 2.0 * ra * k) - rb;
  o.color = color;
  return o;
}

SDF sdfLink(vec3 p, vec3 pos, mat3 rot, float le, float r1, float r2, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 q = vec3(pl.x, max(abs(pl.y) - le, 0.0), pl.z);
  SDF o;
  o.dist = length(vec2(length(q.xy) - r1, q.z)) - r2;
  o.color = color;
  return o;
}

SDF sdfCylinder(vec3 p, vec3 pos, mat3 rot, vec3 c, vec3 color) {
  vec3 pl = rot * (p - pos);
  SDF o;
  o.dist = length(pl.xz - c.xy) - c.z;
  o.color = color;
  return o;
}

SDF sdfCone(vec3 p, vec3 pos, mat3 rot, vec2 c, float h, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec2 q = h * vec2(c.x / c.y, -1.0);
  vec2 w = vec2(length(pl.xz), pl.y);
  vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
  vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
  float k = sign(q.y);
  float d = min(dot(a, a), dot(b, b));
  float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
  SDF o;
  o.dist = sqrt(d) * sign(s);
  o.color = color;
  return o;
}

SDF sdfPlane(vec3 p, vec3 pos, mat3 rot, vec3 n, float h, vec3 color) {
  vec3 pl = rot * (p - pos);
  SDF o;
  o.dist = dot(pl, n) + h;
  o.color = color;
  return o;
}

SDF sdfHexPrism(vec3 p, vec3 pos, mat3 rot, vec2 h, vec3 color) {
  vec3 pl = rot * (p - pos);
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  pl = abs(pl);
  pl.xy -= 2.0 * min(dot(k.xy, pl.xy), 0.0) * k.xy;
  vec2 d = vec2(
      length(pl.xy - vec2(clamp(pl.x, -k.z * h.x, k.z * h.x), h.x)) * sign(pl.y - h.x),
      pl.z - h.y
    );
  SDF o;
  o.dist = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  o.color = color;
  return o;
}

SDF sdfTriPrism(vec3 p, vec3 pos, mat3 rot, vec2 h, vec3 color) {
  vec3 pl = abs(rot * (p - pos));
  SDF o;
  o.dist = max(pl.z - h.y, max(pl.x * 0.866025 + pl.y * 0.5, -pl.y) - h.x * 0.5);
  o.color = color;
  return o;
}

SDF sdfCapsule(vec3 p, vec3 pos, mat3 rot, vec3 a, vec3 b, float r, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec3 pa = pl - a;
  vec3 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  SDF o;
  o.dist = length(pa - ba * h) - r;
  o.color = color;
  return o;
}

SDF sdfVerticalCapsule(vec3 p, vec3 pos, mat3 rot, float h, float r, vec3 color) {
  vec3 pl = rot * (p - pos);
  pl.y -= clamp(pl.y, 0.0, h);
  SDF o;
  o.dist = length(pl) - r;
  o.color = color;
  return o;
}

SDF sdfCappedCylinder(vec3 p, vec3 pos, mat3 rot, float r, float h, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec2 d = abs(vec2(length(pl.xz), pl.y)) - vec2(r, h);
  SDF o;
  o.dist = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  o.color = color;
  return o;
}

SDF sdfRoundedCylinder(vec3 p, vec3 pos, mat3 rot, float ra, float rb, float h, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec2 d = vec2(length(pl.xz) - ra + rb, abs(pl.y) - h + rb);
  SDF o;
  o.dist = min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - rb;
  o.color = color;
  return o;
}

SDF sdfCappedCone(vec3 p, vec3 pos, mat3 rot, float h, float r1, float r2, vec3 color) {
  vec3 pl = rot * (p - pos);
  vec2 q = vec2(length(pl.xz), pl.y);
  vec2 k1 = vec2(r2, h);
  vec2 k2 = vec2(r2 - r1, 2.0 * h);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
  vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  SDF o;
  o.dist = s * sqrt(min(dot(ca, ca), dot(cb, cb)));
  o.color = color;
  return o;
}

SDF sdfRoundCone(vec3 p, vec3 pos, mat3 rot, float r1, float r2, float h, vec3 color) {
  vec3 pl = rot * (p - pos);
  float b = (r1 - r2) / h;
  float a = sqrt(1.0 - b * b);
  vec2 q = vec2(length(pl.xz), pl.y);
  float k = dot(q, vec2(a, b));
  SDF o;
  if (k < 0.0) o.dist = length(q) - r1;
  else if (k > a * h) o.dist = length(q - vec2(0.0, h)) - r2;
  else o.dist = dot(q, vec2(a, b)) - r1;
  o.color = color;
  return o;
}

SDF sdfEllipsoid(vec3 p, vec3 pos, mat3 rot, vec3 r, vec3 color) {
  vec3 pl = rot * (p - pos);
  float k0 = length(pl / r);
  float k1 = length(pl / (r * r));
  SDF o;
  o.dist = k0 * (k0 - 1.0) / k1;
  o.color = color;
  return o;
}

void initRayout(out RayInfo ray)
{
  vec2 uv = (gl_FragCoord.xy / pc.resolution.xy) * 2.0 - 1.0; // [-1,1]
  uv.y = -uv.y;
  uv.x *= pc.resolution.x / pc.resolution.y; // aspect correction
  mat3 camRot = mat3(1.0);
  ray.origin = vec3(-2.0, -2.0, 0.0);

  if (pc.state <= 2) {
    ray.origin = vec3(-2.0, -2.0, 0.0);
    camRot = rotatex(0.7) * rotatey(0.4);
  } else if (pc.state == 3) {
    ray.origin = vec3(1.0, -3.5, 0.0);
    camRot = mat3(1.0);
  } else if (pc.state >= 4 && pc.state < 8) {
    ray.origin = vec3(2.0, -3.5, 2.0);
    camRot = rotatey(-1.6);
  } else if (pc.state == 9) {
    ray.origin = vec3(0.0, 0.0, -2.0);
    camRot = mat3(1.0);
  } else if (pc.state == 10) {
    ray.origin = vec3(0.0, 0.0, -2.0 - smoothstep(0.0, 10.0, pc.time - pc.starttime) * 10.0);
    camRot = mat3(1.0);
  } else if (pc.state >= 11) {
    ray.origin = vec3(0.0, 0.0, -12.0);
    camRot = mat3(1.0);
  }

  // Camera frame
  vec3 forward = normalize(vec3(uv, 1.0)); // camera looks along this
  vec3 worldUp = vec3(0.0, 1.0, 0.0);
  vec3 right = normalize(cross(forward, worldUp));
  vec3 up = cross(right, forward);

  // Field of view
  float fovRad = radians(1.0); // or pass cameraFov as uniform
  float halfHeight = tan(fovRad / 2.0);
  float halfWidth = halfHeight * (pc.resolution.x / pc.resolution.y);

  // Ray in world space
  ray.dir = normalize(forward + uv.x * halfWidth * right + uv.y * halfHeight * up);
  ray.dir *= camRot;
}

SDF map(vec3 p) {
  // Example primitives

  vec3 globalPos = vec3(0.0, 0.0, 0.0);
  //scene 1:
  if (pc.state < 9)
  {
    vec3 roomPos = vec3(0.0, 4.0, 0.0);
    vec3 roomSize = vec3(10.0);
    vec3 roomColor = vec3(1.2, 1.0, 1.0);
    SDF roomGeometry = sdfBox(p, roomPos + globalPos, mat3(1.0), roomSize, roomColor); // blue

    roomSize = vec3(3.0, 9.0, 3.0);
    SDF roomHole = sdfBox(p, roomPos + globalPos, mat3(1.0), roomSize, roomColor);
    roomGeometry = opSubtraction(roomHole, roomGeometry);

    SDF scene = roomGeometry;

    //bed
    vec3 bedPos = vec3(1.5, -5.0, 5.0);
    vec3 bedSize = vec3(1.0, 0.5, 5.0);
    vec3 bedColor = vec3(1.0, 1.0, 1.0) * 2.0;
    SDF bed = sdfRoundBox(p, bedPos + globalPos, mat3(1.0), bedSize, 0.1, bedColor);

    vec3 rimPos = vec3(1.5, -4.5, 1.55);
    SDF bedRim = sdfRoundedBoxFrame(p, rimPos + globalPos, mat3(1.0), vec3(0.95, 0.0, 1.44), 0.0, 0.02, bedColor);
    bed = opSmoothUnion(bed, bedRim, 0.03);
    scene = opUnion(bed, scene);

    //end table
    vec3 endTablePos = vec3(-1.4, -4.0, 3.0);
    vec3 cutoutPos = endTablePos + vec3(0.0, 1.0, 0.0);
    SDF endtable = sdfCappedCylinder(p, endTablePos + globalPos, rotatex(1.6), 1.0, 0.5, roomColor * 1.2);
    SDF cutout = sdfBox(p, cutoutPos + globalPos, mat3(1.0), vec3(1.2), vec3(1.2, 1.0, 1.0));
    endtable = opSubtraction(cutout, endtable);
    scene = opUnion(endtable, scene);

    //mask
    vec3 maskPos = vec3(0.0, -3.5, 2.0);
    maskPos.y += sin(pc.time) * 0.1;
    vec3 maskEllipseSize = vec3(0.3, 0.4, 0.23);
    vec3 maskColor = vec3(1.3, 355.0 / 255.0, 355.0 / 255.0);
    vec3 accentColor = vec3(0.0, 0.0, 0.0);
    SDF maskEllipse1 = sdfEllipsoid(p, vec3(0.0, 0.0, 0.0) + maskPos + globalPos, mat3(1.0), maskEllipseSize, maskColor);
    SDF maskEllipse2 = sdfEllipsoid(p, vec3(-0.13, 0.0, 0.0) + maskPos + globalPos, mat3(1.0), maskEllipseSize, maskColor);

    SDF mask = maskEllipse1;

    vec3 headSize = vec3(0.25, 0.28, 0.2);
    SDF head = sdfEllipsoid(p, vec3(0.1, 0.1, 0.0) + maskPos + globalPos, mat3(1.0), headSize, maskColor);
    mask = opSmoothUnion(head, mask, 0.05);

    vec3 chinSize = vec3(0.25, 0.28, 0.2) - p.x * vec3(0.0, 0.0, 0.1);
    SDF chin = sdfEllipsoid(p, vec3(0.1, -0.1, 0.0) + maskPos + globalPos, mat3(1.0), chinSize, maskColor);
    mask = opSmoothUnion(chin, mask, 0.05);

    vec3 eyeBagSize = vec3(0.03, 0.06, 0.03);
    vec3 mirrorP = p;
    mirrorP.z = abs(mirrorP.z - (maskPos.z + globalPos.z))
        + (maskPos.z + globalPos.z);
    SDF eyeBag = sdfEllipsoid(mirrorP, vec3(0.35, 0.0, 0.1) + maskPos + globalPos, rotatez(-0.3), eyeBagSize, accentColor);
    mask = opSmoothSubtraction(eyeBag, mask, 0.1);

    vec3 eyeHoleSize = vec3(0.06, 0.02, 0.04);
    mirrorP = p;
    mirrorP.z = abs(mirrorP.z - (maskPos.z + globalPos.z))
        + (maskPos.z + globalPos.z);
    SDF eyeHole = sdfEllipsoid(mirrorP, vec3(0.25, 0.05, 0.08) + maskPos + globalPos, rotatez(-0.6), eyeHoleSize, maskColor);
    mask = opSmoothSubtraction(eyeHole, mask, 0.05);

    vec3 noseBridgeSize = vec3(0.02, 0.02, 0.12);
    SDF noseBridge = sdfRoundedCylinder(p, vec3(0.33, 0.0, 0.0) + maskPos + globalPos, rotatez(0.5), noseBridgeSize.x, noseBridgeSize.y, noseBridgeSize.z, maskColor);
    mask = opSmoothUnion(noseBridge, mask, 0.05);

    vec3 nostrilSize = vec3(0.02, 0.02, 0.02);
    mirrorP = p;
    mirrorP.z = abs(mirrorP.z - (maskPos.z + globalPos.z))
        + (maskPos.z + globalPos.z);
    SDF nostril = sdfEllipsoid(mirrorP, vec3(0.35, -0.09, 0.03) + maskPos + globalPos, rotatez(-0.3), nostrilSize, maskColor);
    mask = opSmoothUnion(nostril, mask, 0.02);

    mask = opSmoothSubtraction(maskEllipse2, mask, 0.1);
    scene = opUnion(mask, scene);

    //bed and person
    vec3 blanketPos = vec3(1.5, -4.5, 1.0);
    vec3 blanketSize = vec3(1.0, 0.1, 1.0);
    vec3 blanketColor = bedColor * 0.3;
    SDF blanket = sdfRoundBox(p, blanketPos + globalPos, mat3(1.0), blanketSize, 0.1, blanketColor);
    scene = opSmoothUnion(blanket, scene, 0.1);

    vec3 torsoSize = vec3(0.2, 0.1, 0.4);
    SDF torso = sdfRoundedCylinder(p, vec3(0.0, 0.1, 0.0) + blanketPos + globalPos, rotatez(1.6) * rotatey(1.3), torsoSize.x, torsoSize.y, torsoSize.z, blanketColor);
    scene = opSmoothUnion(torso, scene, 0.1);

    vec3 upperLegSize = vec3(0.1, 0.1, 0.4);
    SDF upperLeg = sdfRoundedCylinder(p, vec3(-0.2, 0.1, -0.2) + blanketPos + globalPos, rotatez(1.6) * rotatey(0.5), upperLegSize.x, upperLegSize.y, upperLegSize.z, blanketColor);
    scene = opSmoothUnion(upperLeg, scene, 0.2);

    vec3 lowerLegSize = vec3(0.1, 0.1, 0.4);
    SDF lowerLeg = sdfRoundedCylinder(p, vec3(-0.4, 0.1, -0.3) + blanketPos + globalPos, rotatez(1.6) * rotatey(1.3), lowerLegSize.x, lowerLegSize.y, lowerLegSize.z, blanketColor);
    scene = opSmoothUnion(lowerLeg, scene, 0.1);

    SDF headInBed = sdfSphere(p, vec3(-0.2, 0.1, 0.5) + blanketPos + globalPos, mat3(1.0), 0.2, blanketColor);
    scene = opSmoothUnion(headInBed, scene, 0.1);
    return scene;
  } else {
    float localtime = pc.time - pc.starttime;
    if (pc.state == 9)
    {
      localtime = 0.0;
    }
    vec3 wallColor = vec3(2.0);
    vec3 backWallPos = vec3(0.0, 0.0, 5.0);
    vec3 backWallSize = vec3(4.0, 2.0, 0.1);
    SDF backWall = sdfBox(p, backWallPos, mat3(1.0), backWallSize, wallColor);

    vec3 sideWallPos = vec3(4.0 + smoothstep(0.0, 10.0, localtime) * 10.0, 0.0, 5.0);
    vec3 sideWallSize = vec3(0.1, 2.0, 2.0);
    vec3 sideWallp = p;
    vec3 pivot = vec3(0.0, 0.0, 5.0);
    sideWallp = pivot + rotatez(localtime / 4.0) * (sideWallp - pivot);
    sideWallp.x = abs(sideWallp.x);
    SDF sideWall = sdfBox(sideWallp, sideWallPos, mat3(1.0), sideWallSize, wallColor);

    vec3 topWallPos = vec3(0.0, 1.9 + smoothstep(0.0, 10.0, localtime) * 10.0, 5.0);
    vec3 topWallSize = vec3(4.0, 0.1, 2.0);
    vec3 topWallp = p;
    topWallp = pivot + rotatez(localtime / 4.0) * (topWallp - pivot);
    SDF topWall = sdfBox(topWallp, topWallPos, mat3(1.0), topWallSize, wallColor);

    vec3 floorPos = vec3(0.0, -1.9, 5.0);
    vec3 floorSize = vec3(4.0, 0.1, 2.0);
    SDF floors = sdfBox(p, floorPos, mat3(1.0), floorSize, wallColor);

    SDF wall = opUnion(backWall, sideWall);
    wall = opUnion(wall, topWall);
    wall = opUnion(wall, floors);
    SDF scene = wall;

    SDF chair;
    vec3 chairLegPos = vec3(0.5, -1.5, 4.0);
    vec3 chairp = p;
    chairp.x = abs(chairp.x);
    chairp.z = abs(chairp.z - 3.8) + 3.8;
    SDF chairLeg = sdfCappedCylinder(chairp, chairLegPos, mat3(1.0), 0.1, 0.5, wallColor);
    chair = chairLeg;

    vec3 seatPos = vec3(0.0, -1.0, 4.0);
    vec3 seatSize = vec3(0.6, 0.1, 0.6);
    SDF seat = sdfBox(p, seatPos, mat3(1.0), seatSize, wallColor);
    chair = opUnion(seat, chair);

    vec3 seatBackPos = vec3(0.0, 0.0, 4.0);
    vec3 seatBackSize = vec3(0.6, 1.0, 0.1);
    SDF seatBack = sdfBox(p, seatBackPos, mat3(1.0), seatBackSize, wallColor);
    chair = opUnion(seatBack, chair);

    scene = opUnion(scene, chair);
    return scene;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// NORMAL FUNCTION //

vec3 normal(in vec3 p, float d) {
  float offset = 0.001;
  vec3 distances = vec3(
      map(p + vec3(offset, 0.0, 0.0)).dist - d,
      map(p + vec3(0.0, offset, 0.0)).dist - d,
      map(p + vec3(0.0, 0.0, offset)).dist - d
    );
  return normalize(distances);
}

///////////////////////////////////////////////////////////////////////////////////////
// LIGHTING FUNCTIONS //

float calcShadow(in vec3 ro, in vec3 rd, float k) {
  float res = 1.0;
  float t = EPSILON + hash(ro) * 0.02;

  for (int i = 0; i < MAX_STEPS && t < MAX_DISTANCE; i++) {
    float h = map(ro + rd * t).dist;
    if (h < MIN_DISTANCE) return 0.0;

    float s = k * h / t;
    res = min(res, s);
    res = mix(res, s, 0.2);

    t += clamp(h, 0.02, 0.25);
  }

  return clamp(res, 0.0, 1.0);
}

float calcOcclusion(vec3 p, vec3 norm) {
  float occ = 0.0;
  float sca = 1.0;
  for (int i = 1; i <= 5; i++) {
    float h = float(i) * 0.02;
    float d = map(p + norm * h).dist;
    occ += (h - d) * sca;
    sca *= 0.5;
  }
  return clamp(1.0 - occ, 0.0, 1.0);
}

void calcLighting(inout vec3 color, in vec3 p, in vec3 norm)
{
  vec3 L = normalize(light.position - p);

  float occ = calcOcclusion(p, norm);
  float sha = calcShadow(p, L, 4.0);

  sha = smoothstep(0.2, 1.0, sha);

  float sunLighting = clamp(dot(norm, L), 0.0, 1.0);
  float skyLighting = clamp(0.5 + 0.5 * norm.y, 0.0, 1.0);

  vec3 indirectDir = normalize(-L * vec3(1.0, 0.0, 1.0));
  float indirectLighting = clamp(dot(norm, indirectDir), 0.0, 1.0);

  vec3 lin = sunLighting * vec3(0.64, 0.67, 0.69)
      * pow(vec3(sha), vec3(1.0, 1.2, 1.5));

  lin += skyLighting * vec3(0.16, 0.20, 0.28) * occ;
  lin += indirectLighting * vec3(0.40, 0.28, 0.20) * occ;

  float distance = length(light.position - p);
  float radius = 6.0 - abs(sin(pc.time * 0.5)) * 0.5;
  if (pc.state >= 8)
  {
    radius += 2.0;
  }

  float attenuation = 1.0 - smoothstep(0.0, radius, distance);

  lin *= attenuation;

  color *= lin;
}

///////////////////////////////////////////////////////////////////////////////////////
// MARCHING FUNCTION //

SDF march(out vec3 p, in RayInfo ray) {
  float distance = 0.0;
  SDF hit;
  for (int i = 0; i < MAX_STEPS && distance < MAX_DISTANCE; i++) {
    p = ray.origin + ray.dir * distance;
    hit = map(p);
    if (hit.dist <= MIN_DISTANCE) return hit;
    distance += hit.dist;
  }
  // Return background SDF
  SDF bg;
  bg.dist = -1.0;
  bg.color = vec3(1.0);
  return bg;
}

///////////////////////////////////////////////////////////////////////////////////////
// DRAW FUNCTION //

void draw(inout vec4 color, in RayInfo ray) {
  vec3 p;
  SDF hit = march(p, ray);
  if (hit.dist != -1.0) {
    vec3 norm = normal(p, hit.dist);
    vec3 col = hit.color;
    calcLighting(col, p, norm);
    color = vec4(col, 1.0);
  } else {
    color = vec4(0.0, 0.0, 0.0, 1.0);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// MAIN //

void main() {
  RayInfo ray;
  vec4 color = vec4(0.0);
  initRayout(ray);
  initLight();
  draw(color, ray);
  outColor = color;
}

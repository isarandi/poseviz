#version 330 core

in vec3 g_normal;
noperspective in vec3 g_bary;

uniform vec3 u_color;
uniform float u_opacity;
uniform mat4 u_view;
uniform vec3 u_edge_color;
uniform float u_edge_width;  // total edge width in pixels

out vec4 frag_color;

void main() {
    vec3 n = normalize(g_normal);

    // Raymond 3-light setup (camera-relative), as in mesh.frag
    mat3 cam_to_world = transpose(mat3(u_view));
    vec3 l0 = cam_to_world * vec3( 0.5,  -0.7071, -0.5);
    vec3 l1 = cam_to_world * vec3(-0.75,  0.5,    -0.433);
    vec3 l2 = cam_to_world * vec3( 0.75,  0.5,    -0.433);

    float diffuse = max(dot(n, l0), 0.0)
                  + max(dot(n, l1), 0.0) * 0.6
                  + max(dot(n, l2), 0.0) * 0.5;
    float intensity = min(0.1 + diffuse, 1.0);

    // Distance to the nearest edge in pixels, via the barycentric gradients;
    // anti-aliased ramp over the edge's half width on each side of it.
    vec3 px = g_bary / fwidth(g_bary);
    float dist = min(min(px.x, px.y), px.z);
    float edge = 1.0 - smoothstep(0.5 * u_edge_width - 0.5, 0.5 * u_edge_width + 0.5, dist);

    frag_color = vec4(mix(u_color * intensity, u_edge_color, edge), u_opacity);
}

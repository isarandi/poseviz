#version 330 core

in vec3 v_normal;
in float v_scalar;

uniform sampler2D u_colormap;
uniform float u_vmin;
uniform float u_vmax;
uniform mat4 u_view;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);

    // Raymond 3-light setup (camera-relative)
    mat3 cam_to_world = transpose(mat3(u_view));
    vec3 l0 = cam_to_world * vec3( 0.5,  -0.7071, -0.5);
    vec3 l1 = cam_to_world * vec3(-0.75,  0.5,    -0.433);
    vec3 l2 = cam_to_world * vec3( 0.75,  0.5,    -0.433);

    float diffuse = max(dot(n, l0), 0.0)
                  + max(dot(n, l1), 0.0) * 0.6
                  + max(dot(n, l2), 0.0) * 0.5;
    float intensity = min(0.1 + diffuse, 1.0);

    // Map scalar to colormap
    float t = clamp((v_scalar - u_vmin) / (u_vmax - u_vmin + 1e-8), 0.0, 1.0);
    vec3 color = texture(u_colormap, vec2(t, 0.5)).rgb;

    frag_color = vec4(color * intensity, 1.0);
}

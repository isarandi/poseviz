#version 330 core

in vec3 in_position;
in vec3 in_normal;
in float in_scalar;

uniform mat4 u_view_proj;

out vec3 v_normal;
out float v_scalar;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_normal = in_normal;
    v_scalar = in_scalar;
}

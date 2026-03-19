#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

uniform mat4 u_view_proj;

out vec3 v_normal;
out vec3 v_color;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_normal = in_normal;
    v_color = in_color;
}

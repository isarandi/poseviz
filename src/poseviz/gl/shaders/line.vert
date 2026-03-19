#version 330 core

in vec3 in_position;

uniform mat4 u_view_proj;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
}

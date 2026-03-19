#version 330 core

in vec3 in_position;
in vec2 in_texcoord;

uniform mat4 u_view_proj;

out vec2 v_texcoord;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_texcoord = in_texcoord;
}

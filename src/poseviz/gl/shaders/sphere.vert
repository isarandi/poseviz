#version 330 core

// Per-vertex attributes (unit sphere)
in vec3 in_position;
in vec3 in_normal;

// Per-instance attributes
in vec3 instance_pos;
in float instance_scale;

uniform mat4 u_view_proj;
uniform vec3 u_color;

out vec3 v_normal;
out vec3 v_color;

void main() {
    vec3 world_pos = in_position * instance_scale + instance_pos;
    gl_Position = u_view_proj * vec4(world_pos, 1.0);
    v_normal = in_normal;
    v_color = u_color;
}

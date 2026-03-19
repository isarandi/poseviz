#version 330 core

// Per-vertex attributes (unit cylinder along Y axis, radius 1, height 1)
in vec3 in_position;
in vec3 in_normal;

// Per-instance attributes
in vec3 instance_start;
in vec3 instance_end;
in float instance_radius;

uniform mat4 u_view_proj;
uniform vec3 u_color;

out vec3 v_normal;
out vec3 v_color;

void main() {
    // Compute cylinder transform
    vec3 axis = instance_end - instance_start;
    float length = max(length(axis), 0.0001);
    vec3 dir = axis / length;

    // Build rotation matrix to align Y axis with dir
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right;
    vec3 forward;

    if (abs(dot(up, dir)) > 0.999) {
        right = vec3(1.0, 0.0, 0.0);
    } else {
        right = normalize(cross(up, dir));
    }
    forward = cross(dir, right);

    mat3 rot = mat3(right, dir, forward);

    // Transform vertex: scale by radius and length, rotate, translate
    vec3 scaled = vec3(in_position.x * instance_radius,
                       in_position.y * length,
                       in_position.z * instance_radius);
    vec3 world_pos = rot * scaled + instance_start;

    gl_Position = u_view_proj * vec4(world_pos, 1.0);
    v_normal = rot * in_normal;
    v_color = u_color;
}

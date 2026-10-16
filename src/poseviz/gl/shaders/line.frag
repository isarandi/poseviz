#version 330 core

uniform vec3 u_color;
// Initialised so that users that never set it (camera wireframes, picking)
// keep drawing fully opaque lines.
uniform float u_opacity = 1.0;

out vec4 frag_color;

void main() {
    frag_color = vec4(u_color, u_opacity);
}

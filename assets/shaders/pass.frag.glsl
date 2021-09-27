#version 450

layout(location = 0) in vec2 in_texCoords;
out vec4 fragColor;

uniform sampler2D u_image;

void main() {
    fragColor = texture2D(u_image, in_texCoords);
}
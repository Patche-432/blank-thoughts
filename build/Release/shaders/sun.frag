#version 330 core
out vec4 FragColor;

void main() {
    // Pure emissive — always full white regardless of lighting
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}

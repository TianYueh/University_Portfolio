#version 330 core

in GS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Color;
} fs_in;

out vec4 FragColor;

uniform sampler2D texture1;

void main()
{
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 ambient = 0.1 * fs_in.Color; // Adjust ambient lighting as needed
    vec3 diffuse = diff * fs_in.Color;
    
    FragColor = texture(texture1, fs_in.TexCoords) * vec4(ambient + diffuse, 1.0);
}

#version 330 core

// TODO:
// Implement Toon shading

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float gloss;
};

struct Light { 
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
	vec3 position;  
};

in vec2 texCoord;
in vec4 worldPos;
in vec3 normal;

uniform Material material;
uniform Light light;
uniform vec3 cameraPos;

out vec4 fragColor;

void main()
{

    vec3 L = normalize(light.position - worldPos.xyz),
         V = normalize(cameraPos - worldPos.xyz),
         R = normalize(reflect(-L, normal)),
         N = normalize(normal);

    float intensity = dot(L, N);

    vec3 low_color = vec3(0.141, 0.0, 0.0);         // Dark brown
    vec3 med_color = vec3(0.518, 0.259, 0.0);      // Brown
    vec3 high_color = vec3(1.0, 0.847, 0.741);    // Skin color

    float low_threshold = 0.0;
    float high_threshold = 0.8;

    vec3 final_color;

    if (intensity < low_threshold) 
        final_color = low_color;
    else if (intensity > high_threshold)
        final_color = high_color;
    else
        final_color = med_color;

    fragColor = vec4(final_color, 1.0);
}

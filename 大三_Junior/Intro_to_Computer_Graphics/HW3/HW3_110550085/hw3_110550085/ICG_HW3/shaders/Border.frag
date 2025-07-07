#version 330 core

// Advanced:
// Implement Border effect

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light { 
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
	vec3 position;  
};

uniform vec3 lightPos;

uniform sampler2D deerTexture;
uniform Material material;
uniform Light light;
uniform vec3 cameraPos;

in vec2 texCoord;
in vec4 worldPos;
in vec3 normal;

out vec4 FragColor;

void main()
{
    vec4 texColor = texture(deerTexture, texCoord);

    vec3 L = normalize(light.position - worldPos.xyz),
		 V = normalize(cameraPos - worldPos.xyz),
		 R = normalize(reflect(-L, normal)),
		 N = normalize(normal);

	if(dot(N, V) < 0.2 && dot(N, V) > -0.2){
		FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
	else{
		FragColor = texColor;
	}
	
}
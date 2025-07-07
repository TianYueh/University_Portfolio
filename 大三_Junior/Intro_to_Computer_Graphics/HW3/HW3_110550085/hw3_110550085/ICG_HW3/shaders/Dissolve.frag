#version 330 core

// Advanced:
// Implement Dissolve effect

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
uniform float dissolveFactor;

in vec2 texCoord;
in vec4 worldPos;
in vec3 normal;
in vec3 aPos_f;

out vec4 FragColor;

void main()
{
    vec4 texColor = texture(deerTexture, texCoord);

    vec3 L = normalize(light.position - worldPos.xyz),
		 V = normalize(cameraPos - worldPos.xyz),
		 R = normalize(reflect(-L, normal)),
		 N = normalize(normal);

	if(aPos_f.x < dissolveFactor){
		discard;
	}
	else{
		FragColor = texColor;
	}
	
}
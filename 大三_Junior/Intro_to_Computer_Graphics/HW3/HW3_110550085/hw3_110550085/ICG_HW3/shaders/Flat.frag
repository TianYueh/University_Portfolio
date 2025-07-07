#version 330 core

// TODO:
// implement Flat shading

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

in vec2 fragTexCoord;
in vec4 fragWorldPos;
in vec3 fragNormal;

out vec4 FragColor;


void main()
{

    vec3 L = normalize(light.position - fragWorldPos.xyz),
		 V = normalize(cameraPos - fragWorldPos.xyz),
		 R = normalize(reflect(-L, fragNormal)),
		 N = normalize(fragNormal);

	vec4 Ka = vec4(material.ambient, 1.0f),
		 Kd = vec4(material.diffuse, 1.0f),
		 Ks = vec4(material.specular, 1.0f),
		 La = vec4(light.ambient, 1.0f),
		 Ld = vec4(light.diffuse, 1.0f),
		 Ls = vec4(light.specular, 1.0f);

    vec4 ambient = La * Ka;
	vec4 diffuse = Ld * Kd * max(dot(L, N), 0.0);
	vec4 specular = Ls * Ks * pow(max(dot(V, R), 0.0), material.shininess);	

    vec4 result = (ambient + diffuse + specular);

    FragColor = texture(deerTexture, fragTexCoord) * result;
}

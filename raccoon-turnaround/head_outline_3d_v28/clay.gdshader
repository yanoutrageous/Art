shader_type spatial;
render_mode unshaded, cull_disabled;

uniform float tone = 0.72;

void fragment() {
	vec3 plane = cross(dFdx(VERTEX), dFdy(VERTEX));
	vec3 normal = normalize(plane * (plane.z < 0.0 ? -1.0 : 1.0));
	float light = max(dot(normal, normalize(vec3(-0.42, 0.65, 0.63))), 0.0);
	ALBEDO = vec3(tone * (0.62 + 0.38 * light));
}

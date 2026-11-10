extends "../head_shape_3d_v27/render_head.gd"

const CLAY = preload("res://clay.gdshader")
const TONES := {"Hat": 0.56, "Fur": 0.80, "EarL": 0.68, "EarR": 0.68, "Scarf": 0.64, "Tail": 0.60}


func add_model(parent: Node3D, path: String) -> Node3D:
	if baseline and path.get_file() == "head_3d.glb":
		path = "res://../head_shape_3d_v27/head_3d.glb"
	return super.add_model(parent, path)


func make_figure(model_path: String) -> Node3D:
	var figure := super.make_figure(model_path)
	if figure == null or OS.get_cmdline_user_args().has("--pieces"):
		return figure
	for part: MeshInstance3D in figure.find_children("*", "MeshInstance3D", true, false):
		var material := ShaderMaterial.new()
		material.shader = CLAY
		material.set_shader_parameter("tone", TONES.get(String(part.name), 0.51))
		part.material_override = material
	return figure

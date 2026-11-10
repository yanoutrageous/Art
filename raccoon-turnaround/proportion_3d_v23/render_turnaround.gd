extends SceneTree

const SIZE := Vector2i(480, 256)
const CAMERA_POSITION := Vector3(0, 26, 330)
const BASE := "res://../walk_3d_v16/"
const TROUSERS := {
	"Near": "res://../walk_3d_v19/near_trousers.glb",
	"Far": "res://../walk_3d_v18/far_trousers.glb",
}

var capture: SubViewport


func _initialize() -> void:
	call_deferred("render")


func render() -> void:
	capture = SubViewport.new()
	capture.size = SIZE
	capture.own_world_3d = true
	capture.transparent_bg = true
	capture.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(capture)

	var camera := Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = SIZE.y
	capture.add_child(camera)
	camera.current = true

	var versions := {"v22": BASE, "v23": "res://"}
	var options := OS.get_cmdline_user_args()
	if options.has("--baseline-only"):
		versions.erase("v23")
	elif options.has("--calibrated-only"):
		versions.erase("v22")
	var step := 90 if options.has("--key-views") else 10
	var part_ids := options.has("--part-ids")
	for version in versions:
		var figure := make_figure(versions[version])
		if figure == null:
			push_error("Turnaround could not load model for " + version)
			quit(1)
			return
		var output := ("res://part_ids/" if part_ids else "res://frames/") + String(version)
		if part_ids:
			paint_part_ids(figure)
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(output))
		for degrees in range(0, 181, step):
			var orbit := -deg_to_rad(degrees)
			camera.position = Basis(Vector3.UP, orbit) * CAMERA_POSITION
			camera.rotation.y = orbit
			await process_frame
			await RenderingServer.frame_post_draw
			var image := capture.get_texture().get_image()
			if image.get_size() != SIZE or image.save_png("%s/%03d.png" % [output, degrees]) != OK:
				push_error("Turnaround could not save %s at %d degrees" % [version, degrees])
				quit(1)
				return
		figure.free()
		print("TURNAROUND_RENDER_OK %s %d views" % [version, range(0, 181, step).size()])
	quit()


func paint_part_ids(figure: Node3D) -> void:
	var colors := {"Fur": Color.RED, "Scarf": Color.GREEN, "Chest": Color.BLUE}
	for node: MeshInstance3D in figure.find_children("*", "MeshInstance3D", true, false):
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.cull_mode = BaseMaterial3D.CULL_DISABLED
		material.albedo_color = colors.get(String(node.name), Color.BLACK)
		node.material_override = material


func make_figure(model_path: String) -> Node3D:
	var figure := Node3D.new()
	capture.add_child(figure)
	var body := add_model(figure, model_path + "body_3d.glb")
	if body == null or add_model(figure, model_path + "head_3d.glb") == null:
		figure.free()
		return null
	for side in TROUSERS:
		for part in ["Thigh", "Shin"]:
			var node := body.find_child(side + part, true, false)
			node.free()
		if add_model(figure, TROUSERS[side]) == null:
			figure.free()
			return null
	return figure


func add_model(parent: Node3D, path: String) -> Node3D:
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	if document.append_from_file(ProjectSettings.globalize_path(path), state) != OK:
		return null
	var model := document.generate_scene(state) as Node3D
	parent.add_child(model)
	for node: MeshInstance3D in model.find_children("*", "MeshInstance3D", true, false):
		for surface in node.mesh.get_surface_count():
			var material := node.mesh.surface_get_material(surface) as BaseMaterial3D
			material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	return model

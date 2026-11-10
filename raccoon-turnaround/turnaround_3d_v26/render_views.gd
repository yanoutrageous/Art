extends "../proportion_3d_v23/render_turnaround.gd"

const STATIC_ANGLES := [0, 45, 90, 135, 180]
const PIECES := ["Hat", "EarL", "EarR", "Fur", "Scarf", "Chest", "Hips",
	"NearArm", "FarArm", "NearGlove", "FarGlove", "NearTrousers", "FarTrousers", "NearBoot", "FarBoot", "Tail"]

var baseline := false
var fitted := false


func render() -> void:
	var options := OS.get_cmdline_user_args()
	baseline = options.has("--baseline")
	fitted = options.has("--mesh-fit")
	var masks := options.has("--pieces")
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
	var path := "res://../proportion_3d_v23/" if baseline else "res://"
	if fitted:
		path = "res://geometry/"
	var figure := make_figure(path)
	if figure == null:
		push_error("Five-view model could not be loaded")
		quit(1)
		return
	if masks:
		paint_piece_ids(figure)

	var folder := "res://diagnostic/baseline" if baseline else "res://frames"
	if masks:
		folder = "res://diagnostic/baseline-ids" if baseline else "res://diagnostic/piece-ids"
		if fitted:
			folder = "res://diagnostic/fit-ids"
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(folder))
	for degrees: int in STATIC_ANGLES:
		var orbit := -deg_to_rad(degrees)
		camera.position = Basis(Vector3.UP, orbit) * CAMERA_POSITION
		camera.rotation.y = orbit
		await process_frame
		await RenderingServer.frame_post_draw
		var image := capture.get_texture().get_image()
		if image.get_size() != SIZE or image.save_png("%s/%03d.png" % [folder, degrees]) != OK:
			push_error("Native static render failed at %s" % degrees)
			quit(1)
			return
	print("V26_FIVE_VIEWS_OK ", folder)
	quit()


func paint_piece_ids(figure: Node3D) -> void:
	for node: MeshInstance3D in figure.find_children("*", "MeshInstance3D", true, false):
		var label := PIECES.find(String(node.name)) + 1
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.cull_mode = BaseMaterial3D.CULL_DISABLED
		material.albedo_color = Color8(label * 14, 255 - label * 11, label * 13)
		node.material_override = material


func add_model(parent: Node3D, path: String) -> Node3D:
	if not baseline and not fitted and path in TROUSERS.values():
		path = "res://" + path.get_file()
	return super.add_model(parent, path)

extends "../turnaround_3d_v26/render_views.gd"


func add_model(parent: Node3D, path: String) -> Node3D:
	var file := path.get_file()
	if file in ["body_3d.glb", "near_trousers.glb", "far_trousers.glb"]:
		path = "res://../turnaround_3d_v26/" + file
	return super.add_model(parent, path)

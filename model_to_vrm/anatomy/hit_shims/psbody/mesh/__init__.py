"""Stand-in for psbody.mesh: HIT inference never builds these; using one raises."""
class Mesh:
    def __init__(self, *a, **k): raise ImportError("psbody.mesh is not installed (viewer only)")
MeshViewer = MeshViewers = Mesh

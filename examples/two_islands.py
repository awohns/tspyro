import pyro
from tspyro.diffusion import ApproximateMatrixExponential
from tspyro.diffusion import make_hex_grid
from tspyro.diffusion import WaypointDiffusion2D


def main():
    bounds = dict(
        east=-2.0,
        west=2.0,
        south=-1.0,
        north=1.0,
    )
    island_center = [(-1.0, 0.0), (1.0, 0.0)]
    island_radius = [0.99, 0.99]
    grid_radius = 0.1

    def on_land(x, y):
        result = False
        for (x0, y0), r in zip(island_center, island_radius):
            result = result | (x - x0) ** 2 + (y - y0) ** 2 < r
        return result

    grid = make_hex_grid(**bounds, radius=grid_radius, predicate=on_land)

    matrix_exp = ApproximateMatrixExponential(
        transition=grid["transition"],
    )

    def model():
        # ...TODO...
        parent_location = "TODO"
        child_location = "TODO"
        time = "TODO"
        pyro.sample(
            "migration",
            WaypointDiffusion2D(
                source=parent_location,
                time=time,
                radius=grid_radius,
                waypoints=grid["waypoints"],
                matrix_exp=matrix_exp,
            ),
            obs=child_location,
        )

    return model  # TODO

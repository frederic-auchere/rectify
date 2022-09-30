# Rectify

Image transformations

## Installation

From the active environment

    python setup.py install

or

    pip install .

or if you want to be able to edit & develop (requires reloading the package)

    pip install -e .


## Examples

Shifts 10 pixels on both axes, rotates 20° around pixel (50, 50) and scale by 1.5:

    from rectify import Rectifier, EuclidianTransform
    import numpy as np

    image = np.ones((100, 100))
    transform = EuclidianTransform(10, 10, 20, 1.5, pivot=(50, 50))
    rectifier = Rectifier(transform)
    transformed = rectifier(image, (100, 100), (0, 100), (0, 100)) 

Composition of a CarringtonTransform with a EuclidianTransform, assuming ``header`` is a WCS compatible FITS header:

    euclidian_transform = EuclidianTransform(0, 0, 20, 1.0, pivot=(50, 50))
    carrington_transform = CarringtonTransform(header)
    transform = euclidian_transform + carrington_transform
    transformed = Rectifier(transform, (100, 100), (45, 90), (10, 55), order=2)

The image is remapped on a 100x100 grid between Carrington longitudes (45°, 90°) and latitudes (10°, 55°). The composition of transformations with the ``+`` operator creates a CompositeTransform object. This allows a single interpolation stage to be used for complex transforms.

import os
import itk
import vtk
import numpy as np
from itk import vtk_image_from_image

PATH = "Data/"

ImageType = itk.Image[itk.F, 3]
TransformType = itk.VersorRigid3DTransform[itk.D]


def load_image(file_name: str) -> ImageType:
    ifr = itk.ImageFileReader[ImageType].New()
    ifr.SetFileName(PATH + file_name)
    ifr.Update()
    return ifr.GetOutput()

def load_image_root(file_name: str) -> ImageType:
    ifr = itk.ImageFileReader[ImageType].New()
    ifr.SetFileName(file_name)
    ifr.Update()
    return ifr.GetOutput()

def itk_image_to_vtk(
    image: ImageType,
    x_start: float = 0.0,
    y_start: float = 0.0,
    x_end: float = 1.0,
    y_end: float = 1.0,
) -> vtk.vtkRenderer:
    vtk_image = itk.vtk_image_from_image(image)

    imageSliceMapper = vtk.vtkImageSliceMapper()
    imageSliceMapper.SetInputData(vtk_image)
    imageSliceMapper.SetSliceAtFocalPoint(True)
    imageSliceMapper.SetOrientation(2)
    imageSliceMapper.SetSliceFacesCamera(True)
    imageSliceMapper.StreamingOn()

    imageSlice = vtk.vtkImageSlice()
    imageSlice.SetMapper(imageSliceMapper)

    renderer = vtk.vtkRenderer()
    renderer.GetActiveCamera().ParallelProjectionOn()
    renderer.AddActor(imageSlice)
    renderer.SetViewport(x_start, y_start, x_end, y_end)
    renderer.ResetCamera()

    return renderer

def register_itk_image(fixed_image: ImageType, moving_image: ImageType) -> ImageType:

    initial_transform = TransformType.New()

    metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
    metric.SetNumberOfHistogramBins(50)

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetNumberOfIterations(200)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetLearningRate(0.75)
    optimizer.SetRelaxationFactor(0.8)
    optimizer.SetGradientMagnitudeTolerance(1e-4)

    centered_transform = itk.CenteredTransformInitializer[
        TransformType, ImageType, ImageType
    ].New()
    centered_transform.SetTransform(initial_transform)
    centered_transform.SetFixedImage(fixed_image)
    centered_transform.SetMovingImage(moving_image)
    centered_transform.GeometryOn()
    centered_transform.InitializeTransform()

    scales = itk.OptimizerParameters[itk.D](initial_transform.GetNumberOfParameters())
    scales.Fill(1.0)

    spacing = fixed_image.GetSpacing()
    translation_scale = 1.0 / (100.0 * min(spacing))
    scales[3] = translation_scale
    scales[4] = translation_scale
    scales[5] = translation_scale
    optimizer.SetScales(scales)

    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
    registration.SetInitialTransform(initial_transform)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetNumberOfLevels(1)

    # Add observer to monitor progress (optional)
    def iteration_update():
        print(
            f"Iteration: {optimizer.GetCurrentIteration()}, "
            f"Metric: {optimizer.GetValue():.6f}"
        )

    optimizer.AddObserver(itk.IterationEvent(), iteration_update)

    registration.Update()

    final_transform = registration.GetTransform()

    return final_transform

def resample_image(
    fixed_image: ImageType, moving_image: ImageType, transform: TransformType
) -> ImageType:
    resample_filter = itk.ResampleImageFilter[ImageType, ImageType].New()
    resample_filter.SetInput(moving_image)
    resample_filter.SetTransform(transform)
    resample_filter.UseReferenceImageOn()
    resample_filter.SetReferenceImage(fixed_image)
    resample_filter.SetDefaultPixelValue(0)

    bspline_interpolator = itk.BSplineInterpolateImageFunction[
        ImageType, itk.D, itk.F
    ].New()

    bspline_interpolator.SetSplineOrder(3)
    resample_filter.SetInterpolator(bspline_interpolator)

    resample_filter.Update()
    return resample_filter.GetOutput()

def get_transform_from_file(
    file_name: str, image_gre1: ImageType, image_gre2: ImageType
) -> TransformType:

    if os.path.exists(file_name):
        reader = itk.TransformFileReaderTemplate[itk.D].New()
        reader.SetFileName(file_name)
        reader.Update()

        transform_list = reader.GetTransformList()
        if transform_list.size() > 0:
            generic_transform = transform_list.front()
            register_transform = itk.VersorRigid3DTransform[itk.D].New()
            register_transform.SetParameters(generic_transform.GetParameters())
            register_transform.SetFixedParameters(
                generic_transform.GetFixedParameters()
            )
        else:
            print("No transform found in file, performing registration...")
            register_transform = register_itk_image(image_gre1, image_gre2)
    else:
        register_transform = register_itk_image(image_gre1, image_gre2)
        writer = itk.TransformFileWriterTemplate[itk.D].New()
        writer.SetInput(register_transform)
        writer.SetFileName(file_name)
        writer.Update()

    return register_transform

def better_visualization(image: ImageType, file_name: str) -> ImageType:

    if os.path.exists(file_name):
        image = itk.imread(file_name)
        return image

    BinaryType = itk.Image[itk.UC, 3]
    thresh_filter = itk.BinaryThresholdImageFilter[ImageType, BinaryType].New()
    thresh_filter.SetInput(image)
    thresh_filter.SetLowerThreshold(1.0)
    thresh_filter.SetUpperThreshold(99999.0)  # max float
    thresh_filter.SetInsideValue(1)
    thresh_filter.SetOutsideValue(0)
    thresh_filter.Update()

    mask = thresh_filter.GetOutput()

    corrector = itk.N4BiasFieldCorrectionImageFilter[
        ImageType, BinaryType, ImageType
    ].New()
    corrector.SetInput(image)
    corrector.SetMaskImage(mask)

    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-4)
    corrector.SetNumberOfFittingLevels(4)

    corrector.Update()
    corrected = corrector.GetOutput()

    smoothed = itk.curvature_flow_image_filter(
        corrected, time_step=0.125, number_of_iterations=5
    )

    arr = itk.GetArrayFromImage(smoothed)
    lower = np.percentile(arr, 1)
    upper = np.percentile(arr, 99)

    # Clamp, then rescale
    arr = np.clip(arr, lower, upper)
    arr = ((arr - lower) / (upper - lower)) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    normalized = itk.GetImageFromArray(arr)
    normalized.CopyInformation(smoothed)

    print("Writing file")

    itk.imwrite(normalized, file_name)

    return normalized


def tumor_segmentation(normalized_image: ImageType, seeds: list[tuple]) -> ImageType:
    BinaryType = itk.Image[itk.UC, 3]
    """cast_filter = itk.CastImageFilter[ImageType, BinaryType].New()
    cast_filter.SetInput(normalized_image)
    cast_filter.Update()
    normalized = cast_filter.GetOutput()"""

    segmenter = itk.ConfidenceConnectedImageFilter[ImageType, BinaryType].New()
    segmenter.SetInput(normalized_image)
    segmenter.SetMultiplier(2.0)
    segmenter.SetNumberOfIterations(5)
    segmenter.SetInitialNeighborhoodRadius(3)
    segmenter.SetReplaceValue(1)
    for seed in seeds:
        segmenter.AddSeed(seed)
    segmenter.Update()
    segmented = segmenter.GetOutput()

    radius = 1

    opened = itk.binary_morphological_opening_image_filter(
        segmented, radius=radius, foreground_value=1
    )

    cleaned = itk.binary_morphological_closing_image_filter(
        opened, radius=radius, foreground_value=1
    )

    rescaler_final = itk.RescaleIntensityImageFilter[BinaryType, ImageType].New()
    rescaler_final.SetInput(cleaned)
    rescaler_final.SetOutputMinimum(0.0)
    rescaler_final.SetOutputMaximum(1.0)
    rescaler_final.Update()
    rescaled_final = rescaler_final.GetOutput()

    return rescaled_final

image_gre1 = load_image("case6_gre1.nrrd")
dim_gre1 = image_gre1.GetImageDimension()
width_gre1, height_gre1, depth_gre1 = image_gre1.GetLargestPossibleRegion().GetSize()

image_gre2 = load_image("case6_gre2.nrrd")
dim_gre2 = image_gre2.GetImageDimension()
width_gre2, height_gre2, depth_gre2 = image_gre2.GetLargestPossibleRegion().GetSize()

normalized_gre1 = better_visualization(image_gre1, "tmp_norm1.nrrd")
normalized_gre2 = better_visualization(image_gre2, "tmp_norm2.nrrd")

NormalizedType = itk.Image[itk.UC, 3]
cast_filter = itk.CastImageFilter[NormalizedType, ImageType].New()
cast_filter.SetInput(normalized_gre1)
cast_filter.Update()
normalized_gre1 = cast_filter.GetOutput()

cast_filter = itk.CastImageFilter[NormalizedType, ImageType].New()
cast_filter.SetInput(normalized_gre2)
cast_filter.Update()
normalized_gre2 = cast_filter.GetOutput()

print("Recallage")
register_transform = get_transform_from_file(
    "recallage2.tfm", normalized_gre1, normalized_gre2
)
normalized_gre2_registered = resample_image(normalized_gre1, normalized_gre2, register_transform)

arr = itk.GetArrayFromImage(normalized_gre2_registered)
arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
arr = ((arr - arr.min()) / (arr.max() - arr.min())) * 255
arr = arr.astype(np.uint8)
image_rescaled = itk.GetImageFromArray(arr)
image_rescaled.CopyInformation(normalized_gre2_registered)

itk.imwrite(image_rescaled, "registered_gre2_normalized.nrrd")
RescaledType = type(image_rescaled)
cast_filter = itk.CastImageFilter[RescaledType, ImageType].New()
cast_filter.SetInput(image_rescaled)
cast_filter.Update()
image_rescaled = cast_filter.GetOutput()

print("Segmentation img 1")
seg1 = tumor_segmentation(
    normalized_gre1, seeds=[(86, 66, 52), (124, 63, 78)] # , (99, 78, 83)
)
itk.imwrite(seg1, "seg1.nrrd")
print("Segmentation img 2")
seg2 = tumor_segmentation(
    image_rescaled, seeds=[(86, 66, 52), (124, 63, 78)] # , (98, 78, 83)
)
itk.imwrite(seg2, "seg2.nrrd")

def compute_volume(seg, spacing):
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    seg_array = itk.GetArrayFromImage(seg)
    tumor_voxels = (seg_array == 1).sum()
    return tumor_voxels * voxel_volume

spacing = seg1.GetSpacing()
vol1 = compute_volume(seg1, spacing)
vol2 = compute_volume(seg2, spacing)
diff = vol2 - vol1
print(f"Volume tumeur 1: {vol1:.2f} mm³")
print(f"Volume tumeur 2: {vol2:.2f} mm³")
print(f"Changement de volume: {diff:.2f} mm³")

a = itk.GetArrayFromImage(seg1)
b = itk.GetArrayFromImage(seg2)

intersection = np.logical_and(a, b).sum()
union = np.logical_or(a, b).sum()
dice = 2. * intersection / (a.sum() + b.sum())
print(f"Dice coefficient: {dice:.4f}")

growth = np.logical_and(b == 1, a == 0).sum()
shrink = np.logical_and(a == 1, b == 0).sum()
print(f"Voxels nouvellement apparus: {growth}")
print(f"Voxels disparus: {shrink}")

diff_array = np.zeros_like(a, dtype=np.uint8)
diff_array[np.logical_and(a == 1, b == 1)] = 1  # intersection
diff_array[np.logical_and(a == 0, b == 1)] = 2  # croissance
diff_array[np.logical_and(a == 1, b == 0)] = 3  # régression

diff_image = itk.GetImageFromArray(diff_array)
diff_image.SetSpacing(seg1.GetSpacing())
diff_image.SetOrigin(seg1.GetOrigin())
diff_image.SetDirection(seg1.GetDirection())

diff_image.SetOrigin(image_gre1.GetOrigin())
diff_image.SetSpacing(image_gre1.GetSpacing())
diff_image.SetDirection(image_gre1.GetDirection())

# Convert label and background to VTK
vtk_labels = vtk_image_from_image(diff_image)
vtk_bg = vtk_image_from_image(image_gre1)

# Isosurface for labels
contour = vtk.vtkDiscreteMarchingCubes()
contour.SetInputData(vtk_labels)
contour.GenerateValues(3, 1, 3)
contour.Update()

mapper_labels = vtk.vtkPolyDataMapper()
mapper_labels.SetInputConnection(contour.GetOutputPort())
mapper_labels.ScalarVisibilityOn()
mapper_labels.SetScalarRange(1, 3)

actor_labels = vtk.vtkActor()
actor_labels.SetMapper(mapper_labels)

# Background volume rendering (semi-transparent)
bg_mapper = vtk.vtkSmartVolumeMapper()
bg_mapper.SetInputData(vtk_bg)

opacity_function = vtk.vtkPiecewiseFunction()
opacity_function.AddPoint(np.min(itk.GetArrayFromImage(image_gre1)), 0.0)
opacity_function.AddPoint(np.max(itk.GetArrayFromImage(image_gre1)), 0.15)

color_function = vtk.vtkColorTransferFunction()
color_function.AddRGBPoint(0,    0.0, 0.0, 0.0)   # noir
color_function.AddRGBPoint(50,   0.2, 0.2, 0.2)
color_function.AddRGBPoint(100,  0.4, 0.4, 0.4)
color_function.AddRGBPoint(150,  0.6, 0.6, 0.6)
color_function.AddRGBPoint(200,  0.8, 0.8, 0.8)
color_function.AddRGBPoint(255,  1.0, 1.0, 1.0)   # blanc

volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_function)
volume_property.SetScalarOpacity(opacity_function)
volume_property.ShadeOff()
volume_property.SetInterpolationTypeToLinear()

bg_volume = vtk.vtkVolume()
bg_volume.SetMapper(bg_mapper)
bg_volume.SetProperty(volume_property)

# Renderer setup
renderer = vtk.vtkRenderer()
renderer.AddActor(actor_labels)
renderer.AddVolume(bg_volume)
renderer.SetBackground(1, 1, 1)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

txt = f"""
Intersection : vert
Croissance : rouge
Régression : bleu

Volume tumeur 1: {vol1:.2f} mm3
Volume tumeur 2: {vol2:.2f} mm3
Changement de volume: {diff:.2f} mm3, {diff/vol1*100:.2f} % d'augmentation
Dice coefficient: {dice:.4f}
Voxels nouvellement apparus: {growth}
Voxels disparus: {shrink}
"""

# Create a text actor
text_actor = vtk.vtkTextActor()
text_actor.SetInput(txt)
text_actor.GetTextProperty().SetFontSize(24)
text_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
text_actor.SetDisplayPosition(10, 10)

# Add the actor to the renderer
renderer.AddActor2D(text_actor)

window.Render()
interactor.Start()


## Python Infrared Imaging Tools

If you're a user of infrared cameras, and have anything between your detector and the object you're trying to measure, you might find something useful here.

_Important: Practically all of the calculations are performed on radiance values, **not** temperature. You will need to know the calibration parameters of your detector to convert between temperature and (apparent) radiance values._

The main aim here is to build a system of objects that (may) influence an infrared measurement, and then find the total impact of this system. This can then be applied to actual measurements to estimate the actual radiance of a target object from the measured radiance. This can then be converted to temperature in Kelvin.

### Simple usage guide: 
1. Find out which objects need to be considered in the imaging path.
2. Collect the properties of the objects. Each object needs a temperature, a background environment temperature, and some combination of emittance, transmittance, and reflectance. Air/Atmosphere needs a length and a relative humidity. Mirrors and Windows may additionally be told what fraction of their reflection is diffuse, as well as if they're aligned with the imaging axis or not.
3. Create a `InfraredObject` (or a subclass) instance for each real object, and collect them in a list, starting with those closest to the detector.
4. Create a `TargetObject` instance with the properties of your imaging target, including background temperature, emittance, and reflectance.
5. Create a `Detector` instance with the temperature and calibration of your detector.
6. Create a `CalibratedRadiance` instance with the radiance measured by the detector.
7. Feed all of these into `find_object_radiance` to get back an estimate of the radiance of your target object.

If you have multiple measurements taken using the same system, then it can be more efficient to use `find_radiance_coefficients` instead, then apply this linear relationship to the measurements manually.

For a detailed description of the physics and full derivation of the calculations used, take a look at /docs/considering_multiple_infrared_objects.pdf.

### Contents
**`DetectorCalibration(object)`**\
A container for the calibration coefficients, along with the methods `to_temperature` and `to_radiance` for conversion.

**`CalibratedRadiance(object)`**\
A handy container for radiance and temperature values, and the conversion between them. Can be created with either of the values (as `float` or `np.ndarry`), and passed a `DetectorCalibration` some later time to allow conversion. Supports simple mathematical operations, too.

**`Detector(object)`**\
Holds the properties of the detector/camera being used, including temperature and calibration in a `DetectorCalibration` instance.

**`TargetObject(object)`**\
Holds the properties of the target object of the measurement, most importantly the emittance.

**`InfraredObject(object)`**\
An object that is along the line of sight between the detector and the target object. Holds information about its emittance and transmittance etc., as well as its temperature and the temperature of the background environment. Has two properties `modification_matrix` and `additional_radiation` that describe the object's impact on incoming radiation and its self emission respectively.

**`Window`, `Mirror`, `Atmosphere`, `Vacuum`**\
Inherit from `InfraredObject`, with modifications and/or simplifications based on the expected properties of these types of object.

**`atmospheric_transmittance(temperature, length, relative_humidity)`**\
Function for estimating the infrared transmittance of a packet of air.

**`system_transfer_matrices(object_system: list)`**\
Combine the modification and self-emission effects from a list of `InfraredObject` into one effective object.

**`find_radiance_coefficients(target_object: TargetObject, detector: Detector, object_system: list)`**\
From a system of `InfraredObject`, a `Detector`, and a `TargetObject`, combine all effects into a pair of linear coefficients that relate measured radiance to the estimated radiance of the target object.

**`find_object_radiance(measured_radiance: CalibratedRadiance, target_object: TargetObject, detector: Detector, object_system: list)`**\
Combine all the information from an infrared imaging system with a measured radiance to give the estimated radiance of the target.

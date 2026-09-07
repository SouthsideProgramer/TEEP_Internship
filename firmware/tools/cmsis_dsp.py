"""
PlatformIO extra_script for env:nano33ble -- put CMSIS-DSP on the build.

The Arduino mbed core ships CMSIS Core but not CMSIS-DSP, and the registry's
`arduino-libraries/Arduino_CMSIS-DSP` package is an empty stub (it carries only
library.properties and a build helper, no sources). So this pulls the official
`platformio/framework-cmsis-dsp` package instead and compiles exactly the one
kernel hls_filter.c calls -- arm_biquad_cascade_df2T_f32 -- rather than the
whole DSP library, which would cost minutes of build time for functions this
firmware never links.
"""
from os.path import join

Import("env")  # noqa: F821  (injected by SCons)

platform = env.PioPlatform()  # noqa: F821
cmsis_dsp_dir = platform.get_package_dir("framework-cmsis-dsp")
assert cmsis_dsp_dir, "framework-cmsis-dsp package missing -- check platform_packages"

env.Append(CPPPATH=[join(cmsis_dsp_dir, "Include"), join(cmsis_dsp_dir, "PrivateInclude")])

env.BuildSources(
    join("$BUILD_DIR", "CMSIS-DSP"),
    join(cmsis_dsp_dir, "Source", "FilteringFunctions"),
    src_filter=["-<*>", "+<arm_biquad_cascade_df2T_f32.c>"],
)

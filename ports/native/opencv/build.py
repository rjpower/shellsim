"""Build upstream OpenCV's array and image-processing libraries first."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(
        ctx,
        configure_args=(
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
            "-DBUILD_LIST=core,imgproc",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_opencv_apps=OFF",
            "-DBUILD_opencv_python2=OFF",
            "-DBUILD_opencv_python3=OFF",
            "-DWITH_IPP=OFF",
            "-DWITH_ITT=OFF",
            "-DWITH_OPENCL=OFF",
            "-DWITH_TBB=OFF",
            "-DWITH_OPENMP=OFF",
            "-DWITH_LAPACK=OFF",
            "-DWITH_EIGEN=OFF",
            "-DWITH_OPENJPEG=OFF",
            "-DWITH_JASPER=OFF",
            "-DWITH_OPENEXR=OFF",
            "-DWITH_JPEG=OFF",
            "-DWITH_PNG=OFF",
            "-DWITH_TIFF=OFF",
            "-DWITH_WEBP=OFF",
            "-DWITH_AVIF=OFF",
            "-DWITH_FFMPEG=OFF",
            "-DWITH_V4L=OFF",
            "-DWITH_GSTREAMER=OFF",
            "-DWITH_GTK=OFF",
            "-DWITH_QT=OFF",
            "-DOPENCV_ENABLE_ALLOCATOR_STATS=OFF",
            "-DCV_TRACE=OFF",
            "-DCV_ENABLE_INTRINSICS=OFF",
            "-DCPU_BASELINE=",
            "-DCPU_DISPATCH=",
        ),
        jobs=ctx.jobs or 64,
    )

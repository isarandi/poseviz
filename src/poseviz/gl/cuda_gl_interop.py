"""CUDA-GL interop for writing CUDA tensors directly to GL textures.

Uses ctypes bindings to libcudart.so — no extra dependencies beyond torch and OpenGL.
The GL texture is registered with CUDA once, then each frame is a device-to-device copy.
"""

import ctypes

_cudart = None

GL_TEXTURE_2D = 0x0DE1
_WRITE_DISCARD = 2  # cudaGraphicsRegisterFlagsWriteDiscard
_D2D = 3  # cudaMemcpyDeviceToDevice


def _get_cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    return _cudart


def _check(err, name):
    if err != 0:
        raise RuntimeError(f"{name} failed with error {err}")


class CudaGLTextureWriter:
    """Manages CUDA-GL interop for zero-copy GPU tensor → GL texture transfer."""

    def __init__(self):
        self._resource = ctypes.c_void_p(0)
        self._registered_glo = None

    def write(self, tensor, texture):
        """Copy a CUDA tensor to a ModernGL texture (GPU-internal).

        Args:
            tensor: (H, W, C) contiguous CUDA tensor, uint8
            texture: moderngl.Texture with matching dimensions
        """
        if not tensor.is_contiguous():
            raise ValueError("CUDA tensor must be contiguous for GL interop")

        cudart = _get_cudart()

        # (Re-)register if texture changed
        if self._registered_glo != texture.glo:
            self._unregister()
            resource = ctypes.c_void_p(0)
            _check(
                cudart.cudaGraphicsGLRegisterImage(
                    ctypes.byref(resource),
                    ctypes.c_uint(texture.glo),
                    GL_TEXTURE_2D,
                    _WRITE_DISCARD,
                ),
                "cudaGraphicsGLRegisterImage",
            )
            self._resource = resource
            self._registered_glo = texture.glo

        # Map GL texture into CUDA address space
        _check(
            cudart.cudaGraphicsMapResources(
                1, ctypes.byref(self._resource), ctypes.c_void_p(0)
            ),
            "cudaGraphicsMapResources",
        )

        try:
            # Get the mapped CUDA array
            cuda_array = ctypes.c_void_p(0)
            _check(
                cudart.cudaGraphicsSubResourceGetMappedArray(
                    ctypes.byref(cuda_array), self._resource, 0, 0
                ),
                "cudaGraphicsSubResourceGetMappedArray",
            )

            # Device-to-device copy: tensor → GL texture
            h, w = tensor.shape[:2]
            c = tensor.shape[2] if tensor.ndim > 2 else 1
            row_bytes = w * c
            _check(
                cudart.cudaMemcpy2DToArray(
                    cuda_array,
                    ctypes.c_size_t(0),  # wOffset
                    ctypes.c_size_t(0),  # hOffset
                    ctypes.c_void_p(tensor.data_ptr()),
                    ctypes.c_size_t(row_bytes),  # src pitch
                    ctypes.c_size_t(row_bytes),  # width in bytes
                    ctypes.c_size_t(h),  # height (rows)
                    _D2D,
                ),
                "cudaMemcpy2DToArray",
            )
        finally:
            cudart.cudaGraphicsUnmapResources(
                1, ctypes.byref(self._resource), ctypes.c_void_p(0)
            )

    def _unregister(self):
        if self._registered_glo is not None:
            _get_cudart().cudaGraphicsUnregisterResource(self._resource)
            self._resource = ctypes.c_void_p(0)
            self._registered_glo = None

    def release(self):
        self._unregister()

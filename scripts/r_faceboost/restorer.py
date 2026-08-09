import os
import sys
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

try:
    import torch.cuda as cuda
except:
    cuda = None

import comfy.utils
import folder_paths
import comfy.model_management as model_management

from scripts.reactor_logger import logger
from r_basicsr.utils.registry import ARCH_REGISTRY
from r_chainner import model_loading
from reactor_utils import (
    tensor2img,
    img2tensor,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face
)


def _find_tensorrt_libs_dir():
    try:
        import tensorrt as trt
        pkg_dir = os.path.dirname(os.path.dirname(trt.__file__))
    except Exception:
        try:
            import site
            pkg_dir = site.getsitepackages()[0]
        except Exception:
            pkg_dir = None
    if pkg_dir:
        p = os.path.join(pkg_dir, "tensorrt_libs")
        if os.path.isdir(p) and any(f.lower().startswith("nvinfer_") for f in os.listdir(p)):
            return p
    return None

_trt_libs_dir = _find_tensorrt_libs_dir()
if _trt_libs_dir and _trt_libs_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _trt_libs_dir + os.pathsep + os.environ.get("PATH", "")


if cuda is not None:
    if cuda.is_available():
        providers = []
        if _trt_libs_dir:
            import onnxruntime as ort
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                trt_cache_path = os.path.join(folder_paths.models_dir, "reactor", ".cache_trt_fp32")
                try:
                    os.makedirs(trt_cache_path, exist_ok=True)
                except Exception:
                    trt_cache_path = None
                trt_options = {
                    "device_id": 0,
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_max_workspace_size": 2147483648,
                }
                if trt_cache_path:
                    trt_options["trt_engine_cache_path"] = trt_cache_path
                providers.append(("TensorrtExecutionProvider", trt_options))
        providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]


def get_restored_face(cropped_face,
                      face_restore_model,
                      face_restore_visibility,
                      codeformer_weight,
                      interpolation: str = "Bicubic"):

    if interpolation == "Bicubic":
        interpolate = cv2.INTER_CUBIC
    elif interpolation == "Bilinear":
        interpolate = cv2.INTER_LINEAR
    elif interpolation == "Nearest":
        interpolate = cv2.INTER_NEAREST
    elif interpolation == "Lanczos":
        interpolate = cv2.INTER_LANCZOS4
    
    face_size = 512
    if "1024" in face_restore_model.lower():
        face_size = 1024
    elif "2048" in face_restore_model.lower():
        face_size = 2048

    scale = face_size / cropped_face.shape[0]
    
    logger.status(f"Boosting the Face with {face_restore_model} | Face Size is set to {face_size} with Scale Factor = {scale} and '{interpolation}' interpolation")

    cropped_face = cv2.resize(cropped_face, (face_size, face_size), interpolation=interpolate)

    # For upscaling the base 128px face, I found bicubic interpolation to be the best compromise targeting antialiasing
    # and detail preservation. Nearest is predictably unusable, Linear produces too much aliasing, and Lanczos produces
    # too many hallucinations and artifacts/fringing.

    model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
    device = model_management.get_torch_device()

    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device=device, dtype=torch.float32)

    try:

        with torch.no_grad():

            from scripts.reactor_swapper import get_face_restore_model_instance
            facerestore_model = get_face_restore_model_instance(face_restore_model, device=device, providers=providers)

            if ".onnx" in face_restore_model:  # ONNX models

                ort_session = facerestore_model
                ort_session_inputs = {}

                for ort_session_input in ort_session.get_inputs():
                    if ort_session_input.name == "input":
                        cropped_face_prep = prepare_cropped_face(cropped_face)
                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                    if ort_session_input.name == "weight":
                        weight = np.array([1], dtype=np.double)
                        ort_session_inputs[ort_session_input.name] = weight

                output = ort_session.run(None, ort_session_inputs)[0][0]
                restored_face = normalize_cropped_face(output)

            else:  # PTH models

                output = facerestore_model(cropped_face_t, w=codeformer_weight)[
                    0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        del output
        torch.cuda.empty_cache()

    except Exception as error:

        print(f"\tFailed inference: {error}", file=sys.stderr)
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

    if face_restore_visibility < 1:
        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

    restored_face = restored_face.astype("uint8")
    return restored_face, scale

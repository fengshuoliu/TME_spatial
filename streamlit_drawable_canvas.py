from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import sys
from pathlib import Path


def _patch_streamlit_image_to_url_compat() -> None:
    try:
        from streamlit.elements import image as st_image  # type: ignore
    except Exception:
        return
    if hasattr(st_image, "image_to_url"):
        return

    image_to_url_fn = None
    try:
        from streamlit.elements.lib.image_utils import image_to_url as image_to_url_fn  # type: ignore
    except Exception:
        try:
            from streamlit.elements.image_utils import image_to_url as image_to_url_fn  # type: ignore
        except Exception:
            image_to_url_fn = None

    if image_to_url_fn is None:
        return

    def _make_layout_config(width):
        if width is None:
            return None

        try:
            import importlib
            for module_name in (
                "streamlit.elements.lib.layout_utils",
                "streamlit.elements.lib.layout_types",
                "streamlit.elements.layout_utils",
                "streamlit.elements.layout_types",
            ):
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    continue
                LayoutConfig = getattr(module, "LayoutConfig", None)
                if LayoutConfig is None:
                    continue
                try:
                    return LayoutConfig(width=width)
                except Exception:
                    try:
                        return LayoutConfig(width=int(width))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            from types import SimpleNamespace
            return SimpleNamespace(width=width)
        except Exception:
            return {"width": width}

    def _compat_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None):
        errors = []
        layout_config = _make_layout_config(width)

        if layout_config is not None:
            try:
                return image_to_url_fn(
                    image=image,
                    layout_config=layout_config,
                    clamp=clamp,
                    channels=channels,
                    output_format=output_format,
                    image_id=image_id,
                )
            except Exception as exc:
                errors.append(exc)

            try:
                return image_to_url_fn(
                    image,
                    layout_config,
                    clamp,
                    channels,
                    output_format,
                    image_id,
                )
            except Exception as exc:
                errors.append(exc)

        try:
            return image_to_url_fn(
                image=image,
                width=width,
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )
        except Exception as exc:
            errors.append(exc)

        try:
            return image_to_url_fn(image, width, clamp, channels, output_format, image_id)
        except Exception as exc:
            errors.append(exc)

        try:
            return image_to_url_fn(
                image=image,
                clamp=clamp,
                channels=channels,
                output_format=output_format,
            )
        except Exception as exc:
            errors.append(exc)
            joined = " | ".join(str(err) for err in errors if str(err).strip())
            raise RuntimeError(
                "Failed to adapt Streamlit image_to_url compatibility call. "
                + joined
            ) from exc

    st_image.image_to_url = _compat_image_to_url


def _candidate_module_files() -> list[Path]:
    this_file = Path(__file__).resolve()
    candidates: list[Path] = []
    seen: set[str] = set()

    for dist_name in ("streamlit-drawable-canvas-fix", "streamlit-drawable-canvas"):
        try:
            dist = importlib_metadata.distribution(dist_name)
        except Exception:
            continue
        for rel in (
            Path("streamlit_drawable_canvas/__init__.py"),
            Path("streamlit_drawable_canvas.py"),
        ):
            try:
                candidate = Path(dist.locate_file(rel)).resolve()
            except Exception:
                continue
            key = str(candidate)
            if candidate.exists() and candidate != this_file and key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _load_module_from_file(module_path: Path):
    module_path = Path(module_path).resolve()
    module_name = f"_streamlit_drawable_canvas_real_{abs(hash(str(module_path)))}"

    if module_path.name == "__init__.py":
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(module_path),
            submodule_search_locations=[str(module_path.parent)],
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError("module is None. This should never happen.")
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_real_module():
    _patch_streamlit_image_to_url_compat()
    errors = []

    for module_path in _candidate_module_files():
        try:
            module = _load_module_from_file(module_path)
            if hasattr(module, "st_canvas"):
                return module
        except Exception as exc:
            errors.append(exc)

    detail = " | ".join(str(exc) for exc in errors if str(exc).strip())
    raise ImportError(
        "Could not import st_canvas from the installed drawable-canvas package. "
        "The local shim searched installed distributions for the real "
        "'streamlit_drawable_canvas' module file. "
        + detail
    )


_real_module = _load_real_module()
st_canvas = _real_module.st_canvas

try:
    CanvasResult = _real_module.CanvasResult
except Exception:
    CanvasResult = None

__all__ = ["st_canvas", "CanvasResult"]

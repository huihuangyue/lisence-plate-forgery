"""整车图片到号牌贴片筛查答案的统一流水线。"""

from .core import SIX_IMAGE_NAMES, analyze_vehicle_image, discover_images, run_batch, run_random_batch

__all__ = ["SIX_IMAGE_NAMES", "analyze_vehicle_image", "discover_images", "run_batch", "run_random_batch"]

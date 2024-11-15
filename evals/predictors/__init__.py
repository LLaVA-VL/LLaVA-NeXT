def get_qwen():
    from .predict_qwen import QwenPredict
    return QwenPredict

def get_llava():
    from .predict_llava import LlavaPredict
    return LlavaPredict

# Export only what's needed
__all__ = ['get_qwen', 'get_llava']
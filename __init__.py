from .RemakerFaceSwap import RemakerFaceSwap

NODE_CLASS_MAPPINGS = {
    'RemakerFaceSwap': RemakerFaceSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'RemakerFaceSwap': 'Remaker Face Swap',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

"""Compatibility utilities for Pydantic v1 and v2."""

def to_dict(model_instance):
    """
    Convert Pydantic model instance to dictionary.
    Works with both Pydantic v1 (.dict()) and v2 (.model_dump()).
    """
    if hasattr(model_instance, 'model_dump'):
        return model_instance.model_dump()
    elif hasattr(model_instance, 'dict'):
        return model_instance.dict()
    else:
        # Fallback to dict conversion
        return dict(model_instance)
from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = 200,
    request_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Generate standard API success response.
    Format:
    {
        "success": true,
        "data": ...,
        "request_id": ...
    }
    """
    content: Dict[str, Any] = {"success": True}
    if data is not None:
        content["data"] = data
    elif message is not None:
        content["data"] = {"message": message}
    else:
        content["data"] = {}

    if request_id:
        content["request_id"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
    )


def error_response(
    code: str,
    message: str,
    status_code: int = 400,
    request_id: Optional[str] = None,
    details: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Generate standard API error response.
    Format:
    {
        "success": false,
        "error": {
            "code": ...,
            "message": ...,
            "request_id": ...
        }
    }
    """
    error_payload: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if request_id:
        error_payload["request_id"] = request_id
    if details is not None:
        error_payload["details"] = details

    content = {
        "success": False,
        "error": error_payload,
    }

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
    )

class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        response.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://www.youtube.com https://www.gstatic.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob: https://img.youtube.com; "
            "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com; "
            "frame-src https://www.youtube.com https://accounts.google.com; "
            "media-src 'self' blob:; "
            "worker-src 'self' blob:;",
        )
        return response

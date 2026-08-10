from urllib.parse import parse_qs, urlparse


def youtube_video_id_from_url(url):
    if not url:
        return ""

    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]

    if hostname == "youtu.be" and path_parts:
        return path_parts[0]

    if hostname in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if path_parts and path_parts[0] in {"shorts", "embed"} and len(path_parts) > 1:
            return path_parts[1]

    return ""

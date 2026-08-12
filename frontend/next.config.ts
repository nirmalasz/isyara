import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const djangoOrigin = process.env.DJANGO_ORIGIN ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${djangoOrigin}/api/:path*`,
      },
      {
        source: "/accounts/:path*",
        destination: `${djangoOrigin}/accounts/:path*`,
      },
    ];
  },
};

export default nextConfig;

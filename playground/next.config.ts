import type { NextConfig } from "next";

// FastAPI (kev.serve) is proxied under /kev so the browser never deals with CORS or ports.
const KEV_API = process.env.KEV_API ?? "http://127.0.0.1:8009";
const VLA_API_4B = process.env.VLA_API_4B ?? process.env.VLA_API ?? "http://127.0.0.1:8014";
const VLA_API_05B = process.env.VLA_API_05B ?? "http://127.0.0.1:8013";

const nextConfig: NextConfig = {
  reactCompiler: true,
  devIndicators: false,
  // Next dev only trusts the hostname it was started with (localhost); without this,
  // opening the app via 127.0.0.1 renders the SSR HTML but never hydrates (no errors, buttons dead).
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    return [
      { source: "/kev/:path*", destination: `${KEV_API}/:path*` },
      { source: "/vla-api/4b/:path*", destination: `${VLA_API_4B}/:path*` },
      { source: "/vla-api/05b/:path*", destination: `${VLA_API_05B}/:path*` },
      // Keep the old path working as an alias for the default 4B model.
      { source: "/vla-api/:path*", destination: `${VLA_API_4B}/:path*` },
    ];
  },
};

export default nextConfig;

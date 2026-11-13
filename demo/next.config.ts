import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      // Multipart requests reach this framework guard before the upload route.
      // Keep room for form headers; the application retains its 8 MB file limit.
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;

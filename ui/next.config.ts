import type { NextConfig } from "next";

// Defaults must match run.sh and src/app/api/tensorboard/route.ts.
const TB_HOST = process.env.TENSORBOARD_HOST || "127.0.0.1";
const TB_PORT = process.env.TENSORBOARD_PORT || "6006";
const TB_PATH_PREFIX = process.env.TENSORBOARD_PATH_PREFIX || "/tensorboard";

const nextConfig: NextConfig = {
  serverExternalPackages: ["better-sqlite3"],
  typescript: {
    ignoreBuildErrors: false,
  },
  devIndicators: false,
  skipTrailingSlashRedirect: true,
  async rewrites() {
    return [
      {
        source: `${TB_PATH_PREFIX}/`,
        destination: `http://${TB_HOST}:${TB_PORT}${TB_PATH_PREFIX}/`,
      },
      {
        source: `${TB_PATH_PREFIX}/:path+`,
        destination: `http://${TB_HOST}:${TB_PORT}${TB_PATH_PREFIX}/:path+`,
      },
    ];
  },
};

export default nextConfig;

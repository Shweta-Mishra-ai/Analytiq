import { defineConfig } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// The backend serves the built frontend (backend/static), as in production.
// Build first: npm run build && rm -rf ../backend/static && cp -r dist ../backend/static
const PORT = 8765
const dataDir = mkdtempSync(join(tmpdir(), 'analytiq-e2e-'))

export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 900 },
    acceptDownloads: true,
    trace: 'retain-on-failure',
    launchOptions: process.env.PW_CHROMIUM_PATH
      ? { executablePath: process.env.PW_CHROMIUM_PATH }
      : {},
  },
  webServer: {
    command: `python -m uvicorn app.main:app --host 127.0.0.1 --port ${PORT}`,
    cwd: '../backend',
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { DATA_DIR: dataDir, APP_ADMIN_KEY: '', APP_PASSWORD: '' },
  },
})

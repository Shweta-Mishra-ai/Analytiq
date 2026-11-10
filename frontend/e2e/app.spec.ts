import { expect, test, type Page } from '@playwright/test'
import { statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const fixture = (name: string) => join(import.meta.dirname, 'fixtures', name)

/** Fails the test on any uncaught page error or server error. */
function watch(page: Page) {
  const problems: string[] = []
  page.on('pageerror', (e) => problems.push(`JS error: ${e.message}`))
  page.on('response', (r) => {
    if (r.url().includes('/api/') && r.status() >= 500) problems.push(`HTTP ${r.status()} ${r.url()}`)
  })
  return problems
}

async function upload(page: Page, file: string) {
  await page.goto('/')
  await page.setInputFiles('input[type=file] >> nth=0', file)
}

test('upload, dashboard charts and every page load without errors', async ({ page }) => {
  const problems = watch(page)
  await upload(page, fixture('sales.csv'))
  await expect(page.getByText('region').first()).toBeVisible()

  await page.click('a[href="/dashboard"]')
  // A Plotly chart actually drew (the partial Plotly build loads and renders).
  await expect(page.locator('.js-plotly-plot .main-svg').first()).toBeVisible()
  await expect(page.locator('.t-metric').first()).toBeVisible()

  for (const path of ['/columns', '/quality', '/insights', '/bi', '/eda', '/deep-analysis',
                      '/segments', '/compare', '/ml', '/ab-test', '/survival', '/reports',
                      '/chat', '/rag']) {
    await page.click(`a[href="${path}"]`)
    await expect(page.locator('main')).not.toBeEmpty()
    await page.waitForLoadState('networkidle')
  }
  expect(problems).toEqual([])
})

test('a report downloads as a real PDF, in English and Hindi', async ({ page }) => {
  const problems = watch(page)
  await upload(page, fixture('sales.csv'))
  await expect(page.getByText('region').first()).toBeVisible()
  await page.click('a[href="/reports"]')
  for (const lang of ['en', 'hi']) {
    await page.selectOption('#report-language', lang)
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 150_000 }),
      page.click('button:has-text("Generate PDF report")'),
    ])
    const path = join(tmpdir(), `report-${lang}.pdf`)
    await download.saveAs(path)
    expect(statSync(path).size).toBeGreaterThan(50_000)
  }
  expect(problems).toEqual([])
})

test('order lines without a revenue column get one, with a scorecard', async ({ page }) => {
  const problems = watch(page)
  await upload(page, fixture('order_lines.csv'))
  await expect(page.getByText('Revenue').first()).toBeVisible()
  await page.click('a[href="/dashboard"]')
  await expect(page.getByText(/revenue/i).first()).toBeVisible()
  expect(problems).toEqual([])
})

test('a program renamed as a CSV is refused with the reason', async ({ page }) => {
  const file = join(tmpdir(), 'renamed_program.csv')
  writeFileSync(file, Buffer.from('4d5a900003000000', 'hex'))
  await upload(page, file)
  await expect(page.getByText(/program or script/i)).toBeVisible()
})

test('insights show what to act on, why revenue moved and a tested forecast', async ({ page }) => {
  const problems = watch(page)
  await upload(page, fixture('sales.csv'))
  await expect(page.getByText('region').first()).toBeVisible()
  await page.click('a[href="/insights"]')
  await expect(page.getByTestId('outlook')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('priority').first()).toBeVisible()
  // The forecast states its tested accuracy beside the naive baseline.
  await expect(page.getByTestId('forecast-accuracy')).toContainText("holding last month")
  expect(problems).toEqual([])
})

test('the account page shows storage and limits, and the upload page the deletion date', async ({ page }) => {
  const problems = watch(page)
  await upload(page, fixture('sales.csv'))
  await expect(page.getByText('region').first()).toBeVisible()
  await expect(page.getByTestId('expires').first()).toContainText('deleted on')
  await page.click('a[href="/account"]')
  await expect(page.getByTestId('usage')).toContainText('MB used')
  await expect(page.getByText('Largest file you can upload')).toBeVisible()
  expect(problems).toEqual([])
})

test('the admin screen opens with the key and shows the server status', async ({ page }) => {
  const problems = watch(page)
  await page.goto('/admin')
  await page.fill('input[type=password]', 'any-key-in-open-mode')
  await page.click('button:has-text("Open")')
  await expect(page.getByTestId('admin')).toBeVisible()
  await expect(page.getByText('Disk free')).toBeVisible()
  await expect(page.getByText('Download backup')).toBeVisible()
  expect(problems).toEqual([])
})

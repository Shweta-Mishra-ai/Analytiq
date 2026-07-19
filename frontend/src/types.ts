export interface Figure {
  data: Record<string, unknown>[]
  layout: Record<string, unknown> & { font?: Record<string, unknown> }
}

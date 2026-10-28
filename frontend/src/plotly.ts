/**
 * The Plotly build this app ships: the core plus the trace types the app
 * actually draws.
 *
 * react-plotly.js's default entry imports `plotly.js/dist/plotly` — every
 * trace type Plotly has (3D, maps, finance, …), unminified — which made
 * the chart chunk 4.6 MB. These seven are every type the API returns and
 * every page builds. `scattergl` is here because Plotly Express switches a
 * scatter or line chart to WebGL above 1,000 points. A trace type missing
 * from this list does not throw: the chart just renders empty, so adding a
 * new chart type means registering it here too.
 */
import Plotly from 'plotly.js/lib/core'
import bar from 'plotly.js/lib/bar'
import box from 'plotly.js/lib/box'
import heatmap from 'plotly.js/lib/heatmap'
import histogram from 'plotly.js/lib/histogram'
import pie from 'plotly.js/lib/pie'
import scattergl from 'plotly.js/lib/scattergl'

// `scatter` is part of the core.
Plotly.register([bar, box, heatmap, histogram, pie, scattergl])

export const REGISTERED_TRACE_TYPES = [
  'scatter', 'bar', 'box', 'heatmap', 'histogram', 'pie', 'scattergl',
] as const

export default Plotly

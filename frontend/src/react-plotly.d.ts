declare module 'plotly.js/lib/core' {
  const Plotly: { register: (modules: unknown[]) => void } & Record<string, unknown>
  export default Plotly
}
declare module 'plotly.js/lib/*' {
  const traceModule: unknown
  export default traceModule
}

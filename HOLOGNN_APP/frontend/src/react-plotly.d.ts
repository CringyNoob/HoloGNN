// Shim for react-plotly.js — overrides @types/react-plotly.js if needed
declare module 'react-plotly.js' {
  import type { Component, CSSProperties } from 'react'

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type AnyObj = Record<string, any>

  interface PlotParams {
    data: AnyObj[]
    layout?: AnyObj
    config?: AnyObj
    frames?: AnyObj[]
    style?: CSSProperties
    className?: string
    useResizeHandler?: boolean
    divId?: string
    revision?: number
    [key: string]: unknown
  }

  export default class Plot extends Component<PlotParams> {}
}

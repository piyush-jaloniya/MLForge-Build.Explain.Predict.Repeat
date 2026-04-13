import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'

interface Props {
  figure: Record<string, unknown>
  height?: number
  style?: React.CSSProperties
}

export default function PlotlyChart({ figure, height = 400, style }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || !figure) return
    const el = ref.current
    const layout = {
      ...(figure.layout as object || {}),
      height,
      margin: { t: 40, r: 20, b: 50, l: 60 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#f9fafb',
    }

    Plotly.react(el, (figure.data as Plotly.Data[]) || [], layout as Plotly.Layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      displaylogo: false,
    }).catch(console.error)

    return () => {
      try { Plotly.purge(el) } catch { }
    }
  }, [figure, height])

  return (
    <div
      ref={ref}
      style={{ width: '100%', minHeight: height, ...style }}
    />
  )
}

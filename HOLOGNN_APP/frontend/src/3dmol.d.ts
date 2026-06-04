declare module '3dmol' {
  export interface ViewerSpec {
    backgroundColor?: string
    width?: number
    height?: number
    antialias?: boolean
    cartoonQuality?: number
  }

  export interface StyleSpec {
    cartoon?: {
      color?: string
      colorfunc?: (atom: AtomSpec) => string
      opacity?: number
    }
    stick?: Record<string, unknown>
    sphere?: Record<string, unknown>
    line?: Record<string, unknown>
    surface?: Record<string, unknown>
  }

  export interface AtomSpec {
    elem?: string
    serial?: number
    resi?: number
    resn?: string
    chain?: string
    b?: number
    x?: number
    y?: number
    z?: number
    [key: string]: unknown
  }

  export interface GLViewer {
    addModel(data: string, format: string): unknown
    setStyle(sel: Record<string, unknown>, style: StyleSpec): void
    zoomTo(): void
    render(): void
    clear(): void
    removeAllModels(): void
    resize(): void
    setSlab(near: number, far: number): void
    zoom(factor: number, animationDuration?: number): void
    rotate(angle: number, axis: string | {x: number, y: number, z: number}): void
    translate(x: number, y: number): void
    setBackgroundColor(color: string, opacity?: number): void
  }

  export function createViewer(
    element: HTMLElement | string,
    config?: ViewerSpec
  ): GLViewer
}

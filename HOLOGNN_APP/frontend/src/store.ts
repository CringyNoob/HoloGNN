import type { DdgResponse, ScanResponse, IdrResponse, CompareResponse } from './types'

/**
 * Module-level store for the latest computed results.
 * Components read/write these directly; the Export tab checks them.
 * We use a simple pub/sub to allow components to subscribe to changes.
 */

type Listener = () => void

interface Store {
  ddg: DdgResponse | null
  scan: ScanResponse & { sequence?: string } | null
  idr: IdrResponse | null
  compare: CompareResponse | null
}

const store: Store = {
  ddg: null,
  scan: null,
  idr: null,
  compare: null,
}

const listeners = new Set<Listener>()

function notify() {
  listeners.forEach((fn) => fn())
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function getStore(): Readonly<Store> {
  return store
}

export function setDdg(result: DdgResponse) {
  store.ddg = result
  notify()
}

export function setScan(result: ScanResponse & { sequence?: string }) {
  store.scan = result
  notify()
}

export function setIdr(result: IdrResponse) {
  store.idr = result
  notify()
}

export function setCompare(result: CompareResponse) {
  store.compare = result
  notify()
}

// crypto.randomUUID() is only defined in a secure context. Served over plain
// http at a LAN address it is undefined, and the class field that called it
// threw during the first render, so the dashboard mounted nothing and showed a
// blank page — on the very setup where someone is most likely to be watching
// it from another machine. These identifiers correlate requests rather than
// keep secrets, so falling back is safe. getRandomValues is not itself
// secure-context gated, so it is still there to fall back to.
export function randomId(): string {
  const source = globalThis.crypto
  if (source?.randomUUID) return source.randomUUID()
  if (source?.getRandomValues) {
    const bytes = source.getRandomValues(new Uint8Array(16))
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const hex = [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('')
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

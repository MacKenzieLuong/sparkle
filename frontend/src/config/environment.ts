export const config = {
  mockMode: import.meta.env.VITE_MOCK_MODE !== 'false',
  stopWord: (import.meta.env.VITE_STOP_WORD || 'stop').trim().toLowerCase(),
}

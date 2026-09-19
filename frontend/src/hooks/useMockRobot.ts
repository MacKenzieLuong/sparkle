import { useReducer } from 'react'
import { initialState, mockReducer } from '../mocks/mockRobot'

export function useMockRobot() {
  const [robot, dispatch] = useReducer(mockReducer, undefined, initialState)
  return { robot, dispatch }
}

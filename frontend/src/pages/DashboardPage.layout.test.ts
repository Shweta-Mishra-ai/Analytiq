/**
 * Where an added tile lands.
 *
 * The dashboard starts with its tiles two per row. Adding one used to
 * pass `x: 0, y: Infinity`, which react-grid-layout reads as "bottom,
 * hard left" — so every chart a user added took a full row to itself,
 * left half the width empty, and made the page a screen taller each
 * time. The grid stopped being a grid at exactly the moment the user
 * started building it.
 */
import { describe, expect, it } from 'vitest'
import { nextSlot } from './DashboardPage'

const tile = (i: string, x: number, y: number, w = 6, h = 5) => ({
  i, x, y, w, h, minW: 3, minH: 3,
})

describe('adding a tile', () => {
  it('starts at the origin on an empty dashboard', () => {
    expect(nextSlot([])('a')).toMatchObject({ x: 0, y: 0, w: 6 })
  })

  it('sits beside the last tile when the bottom row has room', () => {
    const placed = nextSlot([tile('a', 0, 0)])('b')
    expect(placed).toMatchObject({ x: 6, y: 0 })
  })

  it('starts a new row only once the bottom row is full', () => {
    const full = [tile('a', 0, 0), tile('b', 6, 0)]
    expect(nextSlot(full)('c')).toMatchObject({ x: 0, y: 5 })
  })

  it('fills the gap on the bottom row rather than the first gap it finds', () => {
    // Two full rows and a half-empty third: the new tile belongs beside
    // the tile on the third row, not on a fourth.
    const layout = [
      tile('a', 0, 0), tile('b', 6, 0),
      tile('c', 0, 5), tile('d', 6, 5),
      tile('e', 0, 10),
    ]
    expect(nextSlot(layout)('f')).toMatchObject({ x: 6, y: 10 })
  })

  it('never returns Infinity, which is what forced the old behaviour', () => {
    const placed = nextSlot([tile('a', 0, 0)])('b')
    expect(Number.isFinite(placed.y)).toBe(true)
    expect(Number.isFinite(placed.x)).toBe(true)
  })

  it('respects a taller tile when it starts the next row', () => {
    const layout = [tile('a', 0, 0, 6, 8), tile('b', 6, 0, 6, 8)]
    expect(nextSlot(layout)('c')).toMatchObject({ x: 0, y: 8 })
  })
})

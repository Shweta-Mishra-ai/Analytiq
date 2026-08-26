/**
 * These components put numbers in front of a client who will act on
 * them. A mislabelled column or a swapped precision/recall is not a
 * cosmetic bug — it is a wrong recommendation delivered confidently.
 */
import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import {
  DataCaveats,
  DecisionBands,
  Estimates,
  Interactions,
  LeakageNotes,
  ModelSelection,
  NoSignal,
} from './Analysis'
import type {
  DecisionBand,
  Estimate,
  Interaction,
  ModelChoice,
  ModelVerdict,
} from './Analysis'

const band = (over: Partial<DecisionBand> = {}): DecisionBand => ({
  budget_pct: 10,
  n_targeted: 147,
  n_events_caught: 61,
  total_events: 237,
  precision: 41,
  recall: 26,
  lift: 2.4,
  ...over,
})

describe('DecisionBands', () => {
  const bands = [
    band({ budget_pct: 5, n_targeted: 74, n_events_caught: 38, precision: 51, recall: 16, lift: 3.1 }),
    band({ budget_pct: 10 }),
    band({ budget_pct: 20, n_targeted: 294, n_events_caught: 95, precision: 32, recall: 40, lift: 1.9 }),
  ]

  it('renders one row per action budget', () => {
    render(<DecisionBands bands={bands} />)
    expect(screen.getByText('top 5%')).toBeInTheDocument()
    expect(screen.getByText('top 10%')).toBeInTheDocument()
    expect(screen.getByText('top 20%')).toBeInTheDocument()
  })

  it('keeps hit rate and share of events on their own rows', () => {
    // Precision and recall are both percentages and easily transposed;
    // reading them back off the row is the only way to catch a swap.
    render(<DecisionBands bands={bands} />)
    const row = screen.getByText('top 20%').closest('tr')!
    const cells = within(row).getAllByRole('cell').map((c) => c.textContent)
    expect(cells).toEqual([
      'top 20%',
      '294',
      '95 of 237',
      '32%',
      '40%',
      '1.9x',
    ])
  })

  it('summarises the strongest band, not merely the first', () => {
    render(<DecisionBands bands={bands} />)
    const summary = screen.getByText(/times better than choosing at random/)
    expect(summary.textContent).toContain('top 5%')
    expect(summary.textContent).toContain('3.1 times better')
  })

  it('renders nothing at all when there are no bands', () => {
    const { container } = render(<DecisionBands bands={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('NoSignal', () => {
  const verdict: ModelVerdict = {
    usable: false,
    baseline_score: 0.81,
    baseline_strategy: 'always predict No',
    model_score: 0.79,
    metric: 'accuracy',
    lift: -0.02,
    auc: 0.52,
    minority_recall: 0.04,
    reason: 'no ranking ability',
    verdict:
      'No combination of the recorded fields separates leavers from stayers.',
  }

  it('states the finding and says it is a finding, not a gap', () => {
    render(<NoSignal verdict={verdict} />)
    expect(screen.getByText(/No predictive signal found/)).toBeInTheDocument()
    expect(screen.getByText(verdict.verdict)).toBeInTheDocument()
    expect(
      screen.getByText(/This is a finding, not a gap in the analysis/),
    ).toBeInTheDocument()
  })
})

describe('ModelSelection', () => {
  const choice: ModelChoice = {
    name: 'Gradient Boosting',
    auc: 0.784,
    candidates: [
      ['Logistic Regression', 0.741],
      ['Random Forest', 0.762],
      ['Gradient Boosting', 0.784],
    ],
    threshold: 0.31,
    threshold_basis: 'maximises F1 on held-out data',
    calibrated: true,
    calibration_before: 14,
    calibration_after: 3,
    excluded_high_cardinality: [['employee_id', 1470]],
  }

  it('marks the winner and shows every candidate it beat', () => {
    render(<ModelSelection choice={choice} />)
    expect(
      screen.getByText('selected').parentElement!.textContent,
    ).toContain('Gradient Boosting')
    expect(screen.getByText('0.741')).toBeInTheDocument()
    expect(screen.getByText('0.762')).toBeInTheDocument()
    expect(screen.getByText('0.784')).toBeInTheDocument()
  })

  it('explains the non-default threshold instead of just printing it', () => {
    render(<ModelSelection choice={choice} />)
    const line = screen.getByText(/Operating threshold/).parentElement!
    expect(line.textContent).toContain('0.31')
    expect(line.textContent).toContain('maximises F1 on held-out data')
  })

  it('reports calibration in points moved, both before and after', () => {
    render(<ModelSelection choice={choice} />)
    const text = screen.getByText(/Scores are calibrated/).parentElement!
      .textContent!
    expect(text).toContain('14 percentage points')
    expect(text).toContain('they are now 3')
  })

  it('says nothing about calibration when the model was not calibrated', () => {
    render(
      <ModelSelection
        choice={{ ...choice, calibrated: false, calibration_before: null }}
      />,
    )
    expect(screen.queryByText(/Scores are calibrated/)).toBeNull()
  })

  it('names the fields dropped for too many distinct values', () => {
    render(<ModelSelection choice={choice} />)
    expect(
      screen.getByText(/employee_id \(1,470 values\)/),
    ).toBeInTheDocument()
  })

  it('omits the exclusions line when nothing was excluded', () => {
    render(
      <ModelSelection choice={{ ...choice, excluded_high_cardinality: [] }} />,
    )
    expect(screen.queryByText(/Excluded for having too many/)).toBeNull()
  })
})

describe('LeakageNotes', () => {
  it('names each excluded field with the reason it was excluded', () => {
    render(
      <LeakageNotes
        findings={[
          {
            column: 'exit_interview_date',
            separation: 0.99,
            reason: 'is only recorded after the outcome it predicts',
          },
        ]}
      />,
    )
    expect(screen.getByText('exit_interview_date')).toBeInTheDocument()
    expect(
      screen.getByText(/only recorded after the outcome it predicts/),
    ).toBeInTheDocument()
  })

  it('renders nothing when there is no leakage to report', () => {
    const { container } = render(<LeakageNotes findings={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('Estimates', () => {
  const estimates: Estimate[] = [
    {
      column: 'monthly_income',
      statistic: 'mean',
      value: 6502.93,
      ci_low: 6265.11,
      ci_high: 6740.75,
      n: 1470,
    },
  ]

  it('prints the interval alongside the point estimate', () => {
    render(<Estimates estimates={estimates} />)
    const row = screen.getByText('monthly income').closest('tr')!
    const cells = within(row).getAllByRole('cell').map((c) => c.textContent)
    expect(cells).toEqual([
      'monthly income',
      '6,502.93',
      '6,265.11 – 6,740.75',
      '1,470',
    ])
  })

  it('says what the interval means for reading a difference', () => {
    render(<Estimates estimates={estimates} />)
    expect(
      screen.getByText(/smaller than the interval is not evidence/),
    ).toBeInTheDocument()
  })
})

describe('Interactions', () => {
  const interaction: Interaction = {
    metric: 'satisfaction',
    factor: 'overtime',
    moderator: 'department',
    effect_by_level: { Sales: -0.8, Research: 0.6 },
    reverses: true,
    ratio: 2.1,
    description:
      'Overtime lowers satisfaction in Sales but raises it in Research.',
    effect_sd: 0.7,
  }

  it('flags an effect whose direction reverses between groups', () => {
    render(<Interactions interactions={[interaction]} />)
    expect(screen.getByText('direction reverses')).toBeInTheDocument()
    expect(screen.getByText(interaction.description)).toBeInTheDocument()
    expect(screen.getByText('0.7 SD')).toBeInTheDocument()
  })

  it('does not claim a reversal when the effect only differs in size', () => {
    render(
      <Interactions interactions={[{ ...interaction, reverses: false }]} />,
    )
    expect(screen.queryByText('direction reverses')).toBeNull()
  })
})

describe('DataCaveats', () => {
  it('pluralises a single thin group correctly', () => {
    render(
      <DataCaveats
        rare={[{ column: 'department', level: 'Legal', n: 1, share: 0.001 }]}
        imbalance={[]}
      />,
    )
    const text = screen.getByText(/category level/).textContent!
    expect(text).toContain('1 category level hold')
    expect(text).toContain('at 1 row.')
    expect(text).not.toContain('1 rows')
  })

  it('pluralises several thin groups correctly', () => {
    render(
      <DataCaveats
        rare={[
          { column: 'department', level: 'Legal', n: 4, share: 0.003 },
          { column: 'department', level: 'Facilities', n: 6, share: 0.004 },
        ]}
        imbalance={[]}
      />,
    )
    const text = screen.getByText(/category level/).textContent!
    expect(text).toContain('2 category levels')
    expect(text).toContain('at 4 rows.')
  })

  it('shows the imbalance note as written by the engine', () => {
    render(
      <DataCaveats
        rare={[]}
        imbalance={[
          {
            column: 'attrition',
            majority_level: 'No',
            majority_share: 0.84,
            note: 'Attrition is 84% "No", so accuracy alone is misleading.',
          },
        ]}
      />,
    )
    expect(
      screen.getByText('Attrition is 84% "No", so accuracy alone is misleading.'),
    ).toBeInTheDocument()
  })

  it('renders nothing when the data has no caveats', () => {
    const { container } = render(<DataCaveats rare={[]} imbalance={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})

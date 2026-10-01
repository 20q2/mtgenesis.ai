import {
  CardStatus, CardView, EventView, PoolEntryView, PoolSlotView, PoolView, SetCardView, SetView
} from '../models/api.model';

/** Test-only builders for API view objects. */
export function cardView(overrides: Partial<CardView> = {}): CardView {
  return {
    id: 'c-1',
    userId: 'u-1',
    setId: null,
    slot: null,
    replaced: false,
    status: 'queued' as CardStatus,
    error: null,
    textReady: false,
    artReady: false,
    queuePosition: 1,
    etaSeconds: 10,
    card: {
      name: 'Zur\'ka, Élan of Ash', manaCost: '{2}{R}', colors: ['R'], type: 'Creature',
      rarity: 'mythic', cmc: 3, supertype: 'Legendary'
    },
    cardImageUrl: null,
    artImageUrl: null,
    createdAt: '2026-09-28T20:00:00+00:00',
    shared: false,
    ...overrides
  };
}

export function doneCard(overrides: Partial<CardView> = {}): CardView {
  const id = overrides.id ?? 'c-1';
  return cardView({
    status: 'done', textReady: true, artReady: true, queuePosition: null, etaSeconds: null,
    cardImageUrl: `/api/v1/media/cards/${id}.png`, artImageUrl: `/api/v1/media/art/${id}.png`,
    ...overrides
  });
}

export function setCard(overrides: Partial<SetCardView> = {}): SetCardView {
  const { votes = 0, leader = false, tied = false, ...rest } = overrides;
  return { ...doneCard(rest), votes, leader, tied };
}

export function setView(overrides: Partial<SetView> = {}): SetView {
  const id = overrides.id ?? 's-1';
  return {
    id,
    userId: 'u-1',
    username: 'Alice',
    eventId: 'e-1',
    commanderName: 'Zur\'ka, Élan of Ash',
    prompt: 'a fire elemental queen',
    status: 'locked',
    lockedAt: '2026-09-28T21:00:00+00:00',
    cards: [1, 2, 3].map(slot => setCard({ id: `${id}-c${slot}`, setId: id, slot })),
    myVoteCardId: null,
    ...overrides
  };
}

export function eventView(overrides: Partial<EventView> = {}): EventView {
  return {
    id: 'e-1',
    name: 'AI Night #1',
    status: 'open',
    createdAt: '2026-09-28T19:00:00+00:00',
    closedAt: null,
    sets: [setView()],
    ...overrides
  };
}

export function poolEntry(overrides: Partial<PoolEntryView> = {}): PoolEntryView {
  const id = overrides.id ?? 'pe-1';
  return {
    id,
    slotId: 'ps-1',
    cardId: `${id}-card`,
    mine: false,
    username: null,
    card: doneCard({ id: `${id}-card` }),
    power: { estimate: 1, budget: 1.25, verdict: 'fair' },
    gold: 0,
    silver: 0,
    bronze: 0,
    points: 0,
    leader: false,
    tied: false,
    disqualified: false,
    bans: null,
    myMedal: null,
    bannedByMe: false,
    createdAt: '2026-09-29T19:00:00+00:00',
    ...overrides
  };
}

export function poolSlot(overrides: Partial<PoolSlotView> = {}): PoolSlotView {
  return {
    id: 'ps-1',
    position: 1,
    label: 'Red creature',
    colorRule: 'R',
    typeRule: 'creature',
    ruleText: 'Red creature',
    entries: [],
    myMedals: { gold: null, silver: null, bronze: null },
    myEntryId: null,
    ...overrides
  };
}

export function poolView(overrides: Partial<PoolView> = {}): PoolView {
  return {
    id: 'p-1',
    name: 'Knowledge Pool 2026',
    status: 'open',
    maxEntriesPerUser: 2,
    createdAt: '2026-09-29T18:00:00+00:00',
    closedAt: null,
    myEntryCount: 0,
    bansPerPlayer: 2,
    banThreshold: 3,
    myBansLeft: 2,
    slots: [poolSlot()],
    ...overrides
  };
}

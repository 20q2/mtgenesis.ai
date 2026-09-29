/**
 * Card display model and form option constants.
 * Reconstructed 2026-09-28 from usages (the original was never committed).
 */

export enum Rarity {
  COMMON = 'common',
  UNCOMMON = 'uncommon',
  RARE = 'rare',
  MYTHIC = 'mythic'
}

export interface Card {
  name: string;
  manaCost: string;
  supertype?: string;
  type: string;
  subtype?: string;
  colors: string[];
  cmc: number;
  rarity: Rarity;
  artPrompt?: string;
  description?: string;
  flavorText?: string;
  power?: string;
  toughness?: string;
  setCode?: string;
  cardNumber?: string;
  /** Data URL or absolute URL of the raw artwork */
  imageUrl?: string;
  /** Data URL or absolute URL of the fully rendered card */
  cardImageUrl?: string;
}

export interface ColorOption {
  value: string;
  label: string;
  color: string;
}

export const ColorOptions: ColorOption[] = [
  { value: 'W', label: 'White', color: '#f8e7b9' },
  { value: 'U', label: 'Blue', color: '#b3ceea' },
  { value: 'B', label: 'Black', color: '#a69f9d' },
  { value: 'R', label: 'Red', color: '#e49977' },
  { value: 'G', label: 'Green', color: '#a3c095' },
  { value: 'C', label: 'Colorless', color: '#d5d5d5' }
];

export interface LabeledOption {
  value: string;
  label: string;
  /** mana-font glyph classes (e.g. 'ms-creature'), drawn in order before the label. */
  icons?: string[];
}

export const SupertypeOptions: LabeledOption[] = [
  { value: 'Legendary', label: 'Legendary' },
  { value: 'Basic', label: 'Basic' },
  { value: 'Snow', label: 'Snow' },
  { value: 'World', label: 'World' }
];

export const CardTypeOptions: LabeledOption[] = [
  { value: 'Creature', label: 'Creature', icons: ['ms-creature'] },
  { value: 'Instant', label: 'Instant', icons: ['ms-instant'] },
  { value: 'Sorcery', label: 'Sorcery', icons: ['ms-sorcery'] },
  { value: 'Enchantment', label: 'Enchantment', icons: ['ms-enchantment'] },
  { value: 'Artifact', label: 'Artifact', icons: ['ms-artifact'] },
  { value: 'Artifact Creature', label: 'Artifact Creature', icons: ['ms-artifact', 'ms-creature'] },
  { value: 'Enchantment Creature', label: 'Enchantment Creature', icons: ['ms-enchantment', 'ms-creature'] },
  { value: 'Land', label: 'Land', icons: ['ms-land'] },
  { value: 'Planeswalker', label: 'Planeswalker', icons: ['ms-planeswalker'] },
  { value: 'Battle', label: 'Battle', icons: ['ms-battle'] }
];

export const CommonSubtypes: Record<string, string[]> = {
  Creature: ['Human', 'Elf', 'Goblin', 'Dragon', 'Angel', 'Demon', 'Zombie', 'Vampire', 'Wizard', 'Warrior', 'Knight', 'Beast', 'Elemental', 'Spirit', 'Merfolk', 'Sphinx'],
  Artifact: ['Equipment', 'Vehicle', 'Treasure', 'Food', 'Clue'],
  Enchantment: ['Aura', 'Saga', 'Curse', 'Shrine', 'Class'],
  Land: ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Desert', 'Gate'],
  Planeswalker: ['Jace', 'Chandra', 'Liliana', 'Nissa', 'Gideon', 'Ajani'],
  Instant: ['Arcane', 'Adventure'],
  Sorcery: ['Arcane', 'Adventure', 'Lesson'],
  Battle: ['Siege']
};

export const RarityOptions: LabeledOption[] = [
  { value: Rarity.COMMON, label: 'Common' },
  { value: Rarity.UNCOMMON, label: 'Uncommon' },
  { value: Rarity.RARE, label: 'Rare' },
  { value: Rarity.MYTHIC, label: 'Mythic' }
];

export interface ManaSymbol {
  symbol: string;
  description: string;
  color: string;
  textColor?: string;
  icon?: string;
}

export const ManaSymbols: ManaSymbol[] = [
  { symbol: '{W}', description: 'White mana', color: '#f8e7b9', textColor: '#333' },
  { symbol: '{U}', description: 'Blue mana', color: '#b3ceea', textColor: '#333' },
  { symbol: '{B}', description: 'Black mana', color: '#a69f9d', textColor: '#111' },
  { symbol: '{R}', description: 'Red mana', color: '#e49977', textColor: '#333' },
  { symbol: '{G}', description: 'Green mana', color: '#a3c095', textColor: '#333' },
  { symbol: '{C}', description: 'Colorless mana', color: '#d5d5d5', textColor: '#333' },
  { symbol: '{X}', description: 'Variable mana', color: '#e0e0e0', textColor: '#333' },
  { symbol: '{1}', description: 'One generic mana', color: '#e0e0e0', textColor: '#333' },
  { symbol: '{2}', description: 'Two generic mana', color: '#e0e0e0', textColor: '#333' },
  { symbol: '{3}', description: 'Three generic mana', color: '#e0e0e0', textColor: '#333' },
  { symbol: '{4}', description: 'Four generic mana', color: '#e0e0e0', textColor: '#333' },
  { symbol: '{5}', description: 'Five generic mana', color: '#e0e0e0', textColor: '#333' }
];

export function createEmptyCard(): Card {
  return {
    name: '',
    manaCost: '',
    supertype: '',
    type: '',
    subtype: '',
    colors: [],
    cmc: 0,
    rarity: Rarity.COMMON,
    artPrompt: '',
    description: '',
    flavorText: '',
    power: '',
    toughness: '',
    setCode: '',
    cardNumber: ''
  };
}

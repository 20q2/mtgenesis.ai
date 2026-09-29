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
  icon: string;
}

export const SupertypeOptions: LabeledOption[] = [
  { value: 'Legendary', label: 'Legendary', icon: '👑' },
  { value: 'Basic', label: 'Basic', icon: '⛰️' },
  { value: 'Snow', label: 'Snow', icon: '❄️' },
  { value: 'World', label: 'World', icon: '🌍' }
];

export const CardTypeOptions: LabeledOption[] = [
  { value: 'Creature', label: 'Creature', icon: '🐉' },
  { value: 'Instant', label: 'Instant', icon: '⚡' },
  { value: 'Sorcery', label: 'Sorcery', icon: '📜' },
  { value: 'Enchantment', label: 'Enchantment', icon: '✨' },
  { value: 'Artifact', label: 'Artifact', icon: '⚙️' },
  { value: 'Artifact Creature', label: 'Artifact Creature', icon: '🤖' },
  { value: 'Enchantment Creature', label: 'Enchantment Creature', icon: '🦄' },
  { value: 'Land', label: 'Land', icon: '🏔️' },
  { value: 'Planeswalker', label: 'Planeswalker', icon: '🧙' },
  { value: 'Battle', label: 'Battle', icon: '⚔️' }
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
  { value: Rarity.COMMON, label: 'Common', icon: '●' },
  { value: Rarity.UNCOMMON, label: 'Uncommon', icon: '◆' },
  { value: Rarity.RARE, label: 'Rare', icon: '★' },
  { value: Rarity.MYTHIC, label: 'Mythic', icon: '✦' }
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

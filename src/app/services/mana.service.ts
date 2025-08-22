import { Injectable } from '@angular/core';
import { ManaSymbol } from '../models/card.model';

@Injectable({
  providedIn: 'root'
})
export class ManaService {

  constructor() { }

  /**
   * Converts mana cost string to HTML with mana font icons
   * @param manaCost - The mana cost string (e.g., "{2}{W}{U}")
   * @returns HTML string with mana symbols
   */
  getManaSymbolsHtml(manaCost: string): string {
    if (!manaCost) {
      return '';
    }

    // Extract mana symbols from the mana cost string
    const symbols = manaCost.match(/{[^}]+}/g) || [];
    
    return symbols.map(symbol => {
      // Remove the braces to get the symbol content
      const symbolContent = symbol.slice(1, -1).toLowerCase();
      return `<i class="ms ms-${symbolContent}"></i>`;
    }).join('');
  }

  /**
   * Extracts individual mana symbols from a mana cost string
   * @param manaCost - The mana cost string (e.g., "{2}{W}{U}")
   * @returns Array of symbol strings
   */
  extractManaSymbols(manaCost: string): string[] {
    if (!manaCost) {
      return [];
    }
    return manaCost.match(/{[^}]+}/g) || [];
  }

  /**
   * Gets the CSS class name for a mana symbol
   * @param symbol - The mana symbol (e.g., "{W}" or "W")
   * @returns CSS class name for mana font
   */
  getSymbolClass(symbol: string): string {
    // Remove curly braces and convert to lowercase
    const cleanSymbol = symbol.replace(/[{}]/g, '').toLowerCase();
    
    // Handle hybrid symbols (e.g., "w/u" becomes "wu")
    if (cleanSymbol.includes('/')) {
      return cleanSymbol.replace('/', '');
    }
    
    return cleanSymbol;
  }

  /**
   * Calculates the converted mana cost (CMC) from a mana cost string
   * @param manaCost - The mana cost string (e.g., "{2}{W}{U}")
   * @returns The total converted mana cost
   */
  calculateCMC(manaCost: string): number {
    if (!manaCost) return 0;
    
    // Extract all mana symbols from the mana cost string
    const symbols = manaCost.match(/{[^}]+}/g) || [];
    let cmc = 0;
    
    for (const symbol of symbols) {
      const content = symbol.slice(1, -1); // Remove the braces
      
      // Check if it's a number (generic mana)
      if (/^\d+$/.test(content)) {
        cmc += parseInt(content, 10);
      } else {
        // Colored mana symbols, hybrid symbols, etc. count as 1
        cmc += 1;
      }
    }
    
    return cmc;
  }

  /**
   * Extracts colors from a mana cost string
   * @param manaCost - The mana cost string (e.g., "{2}{W}{U}")
   * @returns Array of color letters
   */
  extractColorsFromManaCost(manaCost: string): string[] {
    console.log('ManaService: extractColorsFromManaCost called with:', manaCost);
    
    if (!manaCost) {
      console.log('ManaService: No mana cost provided, returning empty array');
      return [];
    }
    
    const colors: string[] = [];
    
    if (manaCost.includes('{W}') || manaCost.includes('W/')) colors.push('W');
    if (manaCost.includes('{U}') || manaCost.includes('U/')) colors.push('U');
    if (manaCost.includes('{B}') || manaCost.includes('B/')) colors.push('B');
    if (manaCost.includes('{R}') || manaCost.includes('R/')) colors.push('R');
    if (manaCost.includes('{G}') || manaCost.includes('G/')) colors.push('G');
    
    console.log('ManaService: Extracted colors:', colors);
    return colors;
  }

  /**
   * Formats a mana symbol for display in buttons
   * @param manaSymbol - ManaSymbol object from the model
   * @returns Object with display properties
   */
  formatSymbolForButton(manaSymbol: ManaSymbol) {
    return {
      symbol: manaSymbol.symbol,
      cssClass: this.getSymbolClass(manaSymbol.symbol),
      icon: manaSymbol.icon,
      description: manaSymbol.description,
      color: manaSymbol.color,
      textColor: manaSymbol.textColor || '#333'
    };
  }

  /**
   * Reorders mana symbols according to Magic's standard ordering
   * Order: Numbers/X, W, U, B, R, G
   * Hybrid symbols follow the leftmost color in their ordering
   * @param manaCost - The mana cost string to reorder
   * @returns Reordered mana cost string
   */
  reorderManaSymbols(manaCost: string): string {
    if (!manaCost) return '';

    // Extract all mana symbols
    const symbols = manaCost.match(/{[^}]+}/g) || [];
    
    // Define ordering priority
    const orderPriority = new Map<string, number>();
    
    // Numbers and X come first (0-99)
    for (let i = 0; i <= 99; i++) {
      orderPriority.set(`{${i}}`, i);
    }
    orderPriority.set('{X}', 100);
    
    // Colors: W, U, B, R, G (101-105)
    orderPriority.set('{W}', 101);
    orderPriority.set('{U}', 102);
    orderPriority.set('{B}', 103);
    orderPriority.set('{R}', 104);
    orderPriority.set('{G}', 105);
    
    // Hybrid symbols (200+ range, ordered by leftmost color)
    const hybridSymbols = [
      '{W/U}', '{W/B}', '{U/B}', '{U/R}', '{B/R}', 
      '{B/G}', '{R/G}', '{R/W}', '{G/W}', '{G/U}'
    ];
    hybridSymbols.forEach((symbol, index) => {
      orderPriority.set(symbol, 200 + index);
    });
    
    // Phyrexian mana (300+ range)
    const phyrexianSymbols = ['{W/P}', '{U/P}', '{B/P}', '{R/P}', '{G/P}'];
    phyrexianSymbols.forEach((symbol, index) => {
      orderPriority.set(symbol, 300 + index);
    });
    
    // Sort symbols by priority
    symbols.sort((a, b) => {
      const priorityA = orderPriority.get(a) ?? 999;
      const priorityB = orderPriority.get(b) ?? 999;
      return priorityA - priorityB;
    });
    
    return symbols.join('');
  }
}


import { Component, Input } from '@angular/core';
import { Card, ColorOptions } from '../../models/card.model';
import { ManaService } from '../../services/mana.service';

@Component({
  selector: 'app-card-preview',
  templateUrl: './card-preview.component.html',
  styleUrls: ['./card-preview.component.scss']
})
export class CardPreviewComponent {
  @Input() card: Card = {
    name: 'Card Name',
    manaCost: '{0}',
    type: 'Card Type',
    colors: ['C'],
    cmc: 0,
    rarity: 'common' as any,
    artPrompt: 'Fantasy art of a card',
    description: 'Card text appears here'
  };

  // Default placeholder image - using a data URL to avoid external requests
  placeholderImage = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjY1IiBoZWlnaHQ9IjM3MCIgdmlld0JveD0iMCAwIDI2NSAzNzAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyNjUiIGhlaWdodD0iMzcwIiBmaWxsPSIjMmEyYTJhIi8+Cjx0ZXh0IHg9IjUwJSIgeT0iNDUlIiBkb21pbmFudC1iYXNlbGluZT0ibWlkZGxlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZmZmIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPkNhcmQgQXJ0PC90ZXh0Pgo8dGV4dCB4PSI1MCUiIHk9IjU1JSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjEyIj5XaWxsIEFwcGVhciBIZXJlPC90ZXh0Pgo8L3N2Zz4K';

  constructor(private manaService: ManaService) {}

  getCardBackground(): string {
    if (!this.card.colors || this.card.colors.length === 0) {
      return '#d5d5d5'; // Default to colorless
    }
    
    if (this.card.colors.length === 1) {
      const colorObj = ColorOptions.find(c => c.value === this.card.colors[0]);
      return colorObj ? colorObj.color : '#d5d5d5';
    }
    
    // For multicolor cards
    if (this.card.colors.includes('G') && this.card.colors.includes('W')) {
      return 'linear-gradient(135deg, #a3c095, #f8e7b9)';
    } else if (this.card.colors.includes('W') && this.card.colors.includes('U')) {
      return 'linear-gradient(135deg, #f8e7b9, #b3ceea)';
    } else if (this.card.colors.includes('U') && this.card.colors.includes('B')) {
      return 'linear-gradient(135deg, #b3ceea, #a69f9d)';
    } else if (this.card.colors.includes('B') && this.card.colors.includes('R')) {
      return 'linear-gradient(135deg, #a69f9d, #e49977)';
    } else if (this.card.colors.includes('R') && this.card.colors.includes('G')) {
      return 'linear-gradient(135deg, #e49977, #a3c095)';
    } else {
      // Fallback for other multicolor combinations
      const colorStops = this.card.colors.map((color, index) => {
        const colorObj = ColorOptions.find(c => c.value === color);
        const percentage = (index * 100) / (this.card.colors.length - 1);
        return `${colorObj?.color || '#d5d5d5'} ${percentage}%`;
      }).join(', ');
      
      return `linear-gradient(135deg, ${colorStops})`;
    }
  }

  private calculateCMC(manaCost: string): number {
    return this.manaService.calculateCMC(manaCost);
  }

  getManaSymbols(): string {
    const cmc = this.calculateCMC(this.card.manaCost);
    if (cmc === 0) return '';
    
    let symbols = '';
    const colorSymbols = this.card.colors.map(color => `{${color}}`);
    
    // Create colored mana symbols based on card colors
    for (let i = 0; i < Math.min(cmc, colorSymbols.length); i++) {
      symbols += colorSymbols[i];
    }
    
    // Add generic mana symbols if cmc > number of colors
    const remaining = cmc - colorSymbols.length;
    if (remaining > 0) {
      symbols = `{${remaining}}` + symbols;
    }
    
    return symbols;
  }

  getManaSymbolsHtml(): string {
    return this.manaService.getManaSymbolsHtml(this.card.manaCost);
  }

  getFullTypeLine(): string {
    const supertype = this.card.supertype;
    const type = this.card.type || 'Card Type';
    const subtype = this.card.subtype;
    
    let typeLine = '';
    
    // Add supertype if present
    if (supertype && supertype.trim()) {
      typeLine += supertype + ' ';
    }
    
    // Add main type
    typeLine += type;
    
    // Add subtype if present
    if (subtype && subtype.trim()) {
      typeLine += ` — ${subtype}`;
    }
    
    return typeLine;
  }

  shouldShowPowerToughness(): boolean {
    // Show power/toughness for creatures and vehicles
    const type = this.card.type?.toLowerCase() || '';
    const subtype = this.card.subtype?.toLowerCase() || '';
    
    return type.includes('creature') || 
           subtype.includes('vehicle') ||
           (this.card.power !== undefined && this.card.power !== '') ||
           (this.card.toughness !== undefined && this.card.toughness !== '');
  }
}
